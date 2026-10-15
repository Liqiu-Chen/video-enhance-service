"""RIFE 插帧推理后端 —— 支持 torch / onnx 两条执行路径。

性能要点（按收益从大到小）：
1. **fp16**：修掉 `warplayer.py` 里 warp 网格的 dtype bug 之后可用。
   （原来网格固定 fp32，跑 fp16 会 `expected scalar type Half but found Float`，
   这就是之前判定"RIFE 不能用 fp16"的真正原因。）
2. **批量化**：模型只有 3.04M 参数，在 2880x5120 下单帧调用 GPU 严重吃不饱；
   一次送 N 对能显著提高吞吐。
3. **CUDA Graph**：把整段前向连同 kernel launch 一次性录下来，消除每次调用的
   启动开销。RIFE 的前向有上百个小 kernel，launch 开销占比很高。
4. **ONNX Runtime**：固定 shape 的图，CUDA EP 下算子融合得更好。

注意：所有路径都要求输入尺寸**已经补齐到 32 的倍数**（由调用方保证），
这样 shape 恒定，CUDA Graph 才能录制、ONNX 也只需导出一份。
"""
from __future__ import annotations

import os
import sys

import torch

from config import CACHE, RIFE_SRC, RIFE_WEIGHT, ensure_dirs

if RIFE_SRC not in sys.path:
    sys.path.insert(0, RIFE_SRC)

PAD = 32


def pad_to_multiple(t: torch.Tensor, m: int = PAD):
    """把 (N,C,H,W) 的 H/W 用 replicate 补齐到 m 的倍数，返回 (补齐后张量, (h,w))。"""
    h, w = t.shape[-2], t.shape[-1]
    ph = (h + m - 1) // m * m
    pw = (w + m - 1) // m * m
    if ph == h and pw == w:
        return t, (h, w)
    return torch.nn.functional.pad(t, (0, pw - w, 0, ph - h), mode="replicate"), (h, w)


class _RifeExport(torch.nn.Module):
    """导出用包装：输入 6 通道（两帧 RGB 拼一起），只输出中间帧。

    原始 IFNet 返回 (flow_list, mask, merged_list)，中间量对 ONNX 毫无意义，
    砍掉后图更小、优化器更容易融合。
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):                                  # x: (N, 6, H, W)
        _, _, merged = self.net(x, scale_list=[4.0, 2.0, 1.0])
        return merged[2]


class _TrtBackend:
    """RIFE 的 TensorRT 后端（TRT 10.x 的 tensor-address API）。

    为什么要单独封装：torch 的 CUDA Graph 在本模型上不可用（录制报错），
    而 TRT 是**离线编译**——按固定 shape 把图编译成引擎，运行时只做
    kernel 调度，没有 Python/框架层的逐算子开销。

    实测（960x1696，真实视频帧）：
        精度：TRT-fp16 vs Torch-fp32 = 61.2 dB（远超 35dB 可用线）
              TRT-fp32 vs Torch-fp32 = 107.6 dB（几乎精确）
        速度：TRT-fp16 10.7 ms/对  vs  Torch fp16+warp32 26.4 ms/对 = 2.47x
    """

    def __init__(self, engine_path: str, verbose: bool = True):
        import tensorrt as trt                                  # noqa: PLC0415
        self.trt = trt
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(
                trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT 引擎反序列化失败: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if
                            self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if
                             self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        _m = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        # ⚠️ 输入 dtype 也要读。只读输出会把 fp16 缓冲区的地址喂给期待 fp32 的
        # 引擎，引擎按 fp32 解引用 → 读到垃圾 → PSNR 掉到 5dB，看起来像精度崩了。
        self.in_dtype = _m[self.engine.get_tensor_dtype(self.in_name)]
        self.out_dtype = _m[self.engine.get_tensor_dtype(self.out_name)]
        if verbose:
            print(f"  [rife] TensorRT 引擎就绪 {os.path.basename(engine_path)} "
                  f"in={self.in_dtype} out={self.out_dtype}", flush=True)

    @torch.inference_mode()
    def run(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.in_dtype:
            x = x.to(self.in_dtype)
        x = x.contiguous()
        self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        out = torch.empty(tuple(self.ctx.get_tensor_shape(self.out_name)),
                          dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 失败")
        return out


class RifeEngine:
    """统一的 RIFE 推理接口：`infer(a, b)` 输入两批帧，输出中间帧。"""

    PAD = PAD                       # 暴露成类属性，调用方补尺寸时要用

    def __init__(self, weight: str = RIFE_WEIGHT, backend: str = "torch",
                 fp16: bool = True, batch: int = 2, use_cuda_graph: bool = False,
                 device: str = "cuda", onnx_path: str | None = None,
                 onnx_fp16: bool = True, warp_fp32: bool = True,
                 trt_path: str | None = None, trt_mode: str = "fp16",
                 verbose: bool = True):
        # batch 默认 2 而不是更大：5K 尺寸下 batch=4 就会爆显存，
        # 而且实测 batch 2 相对 1 并没有吞吐收益（模型太小、算力已饱和），
        # 只有显存开销翻倍。真正需要压显存时请开 use_cuda_graph。
        self.weight = weight
        self.backend = backend
        self.fp16 = bool(fp16)
        self.batch = max(1, int(batch))
        self.use_cuda_graph = bool(use_cuda_graph)
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.onnx_path = onnx_path
        self.onnx_fp16 = onnx_fp16
        self.warp_fp32 = bool(warp_fp32)
        self.trt_path = trt_path
        self.trt_mode = trt_mode
        self.trt = None
        self.verbose = verbose

        self.net = None
        self.graph = None
        self.session = None
        self.io_binding = None
        self.H = self.W = 0
        self._built = False
        self._log("加载 RIFE 权重")
        self._load_net()

    # ------------------------------------------------------------------ #
    def _log(self, *a):
        if self.verbose:
            print("  [rife]", *a, flush=True)

    def _load_net(self):
        from rife_ms.IFNet_HDv3 import IFNet                 # noqa: PLC0415
        from rife_ms import warplayer                        # noqa: PLC0415
        # 5K 尺寸下必须把网格采样放回 fp32（原因见 warplayer.py 顶部注释）
        warplayer.FORCE_FP32_WARP = bool(self.fp16 and self.warp_fp32)
        net = IFNet()
        sd = torch.load(self.weight, map_location="cpu")
        # 与 RIFEModel.load_model 的 convert() 一致：只保留带 module. 前缀的键
        sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
        net.load_state_dict(sd)
        net = net.eval().to(self.device)
        if self.fp16:
            net = net.half()
        self.net = net                                    # 原始 IFNet
        self.n_params = sum(p.numel() for p in net.parameters())
        # 统一包一层：原始 forward 返回 (flow_list, mask, merged_list)，
        # 包装后只吐中间帧，torch 与 onnx 两条路径的接口才一致
        self.net = _RifeExport(net)

    # ------------------------------------------------------------------ #
    def build(self, h: int, w: int):
        """按固定的 (H, W)（必须是 32 的倍数）把后端准备好。

        ⚠️ **这里的提前返回条件以前是错的**：写成 `and (self.graph or self.session)`，
        而 torch 后端（默认路径）既不建 graph 也不建 session，两者恒为 None ——
        于是**每次调用都会重建后端并白跑 2 次全尺寸预热前向**。
        流水线每个微批都会调一次 build()，5K 下每次白烧约 1 秒。
        现在用显式的 `_built` 标志判断。
        """
        assert h % PAD == 0 and w % PAD == 0, f"尺寸必须是 {PAD} 的倍数: {h}x{w}"
        if self._built and (self.H, self.W) == (h, w):
            return self
        self.H, self.W = h, w
        self.graph = None
        self.session = None
        self._built = False
        if self.backend == "torch":
            self._build_torch(h, w)
        elif self.backend == "onnx":
            self._build_onnx(h, w)
        elif self.backend == "trt":
            self._build_trt(h, w)
        else:
            raise ValueError(f"未知 backend: {self.backend}")
        return self

    def _build_torch(self, h, w):
        dev, dt = self.device, self.dtype
        B = self.batch
        if not self.use_cuda_graph:
            x = torch.zeros(B, 6, h, w, dtype=dt, device=dev)
            with torch.inference_mode():
                for _ in range(2):
                    self.net(x)
            self._built = True
            self._log(f"torch 后端就绪 {h}x{w} fp16={self.fp16} batch={B}")
            return
        # CUDA Graph：必须用固定的静态输入缓冲区
        static_in = torch.zeros(B, 6, h, w, dtype=dt, device=dev)
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):                     # 预热，触发 cudnn 算法选择
                    self.net(static_in)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = self.net(static_in)
        self.graph = g
        self.static_in = static_in
        self.static_out = static_out
        self._built = True
        self._log(f"torch+CUDA Graph 就绪 {h}x{w} fp16={self.fp16} batch={B}")

    def _build_trt(self, h, w):
        """加载（必要时构建）TensorRT 引擎。"""
        path = self.trt_path or os.path.join(
            CACHE, f"rife_{h}x{w}_{self.trt_mode}.trt")
        if not os.path.exists(path):
            onnx_p = os.path.join(CACHE, f"rife_{h}x{w}.onnx")
            if not os.path.exists(onnx_p):
                export_rife_onnx(self.weight, onnx_p, (h, w), fp16=False,
                                 verbose=self.verbose)
            # 延迟导入：build_trt_rife 是个脚本，别让它变成引擎的硬依赖
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            from build_trt_rife import build_engine              # noqa: PLC0415
            build_engine(onnx_p, path, mode=self.trt_mode, verbose=self.verbose)
        self.trt = _TrtBackend(path, verbose=self.verbose)
        self._built = True

    def _build_onnx(self, h, w):
        import onnxruntime as ort                             # noqa: PLC0415
        ensure_dirs()
        path = self.onnx_path or os.path.join(
            CACHE, f"rife_{h}x{w}{'_fp16' if self.onnx_fp16 else ''}.onnx")
        if not os.path.exists(path):
            export_rife_onnx(self.weight, path, (h, w), fp16=self.onnx_fp16)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provs = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, so, providers=provs)
        # 转 fp16 时用了 keep_io_types=True，所以图内是 fp16、但出入口仍是 fp32。
        # 直接喂 fp16 会被拒（Unexpected input data type），这里按图的实际 I/O 类型适配。
        self.onnx_in_fp16 = "float16" in self.session.get_inputs()[0].type
        self.onnx_out_fp16 = "float16" in self.session.get_outputs()[0].type
        self._built = True
        self._log(f"ONNX 后端就绪 {h}x{w} in={self.session.get_inputs()[0].type} "
                  f"out={self.session.get_outputs()[0].type} "
                  f"providers={self.session.get_providers()}")
        try:                                                   # 零拷贝绑定
            self.io_binding = self.session.io_binding()
            self._log("IOBinding 可用（显存直通，不走主机内存）")
        except Exception:                                      # noqa: BLE001
            self.io_binding = None

    # ------------------------------------------------------------------ #
    def infer(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a/b: (N,3,H,W)，H/W 必须等于 build 时给的尺寸。返回 (N,3,H,W)。"""
        n = a.shape[0]
        assert a.shape[-2:] == (self.H, self.W), \
            f"输入尺寸 {tuple(a.shape[-2:])} 与 build 的 {(self.H, self.W)} 不符"
        if n != self.batch:
            return self._infer_sliced(a, b)
        return self._infer_exact(a, b)

    def _infer_sliced(self, a, b):
        """数量不足一个 batch 时：补齐 -> 一次前向 -> 切片，避免尾批退化成逐对推理。"""
        n = a.shape[0]
        pa = a.new_empty(self.batch, *a.shape[1:])
        pb = b.new_empty(self.batch, *b.shape[1:])
        pa[:n].copy_(a)
        pa[n:].copy_(a[n - 1:n].expand(self.batch - n, *a.shape[1:]))
        pb[:n].copy_(b)
        pb[n:].copy_(b[n - 1:n].expand(self.batch - n, *b.shape[1:]))
        return self._infer_exact(pa, pb)[:n]

    def _infer_exact(self, a, b):
        if self.backend == "trt":
            return self.trt.run(torch.cat([a, b], 1))
        if self.backend == "torch":
            x = torch.cat([a, b], 1)
            if self.graph is not None:
                self.static_in.copy_(x)
                self.graph.replay()
                return self.static_out
            with torch.inference_mode():
                return self.net(x)
        # ---- ONNX ----
        x = torch.cat([a, b], 1)
        want = torch.float16 if self.onnx_in_fp16 else torch.float32
        if x.dtype != want:
            x = x.to(want)
        if self.io_binding is not None:
            dev = "cuda" if self.device.type == "cuda" else "cpu"
            iob = self.session.io_binding()
            iob.bind_input("x", device_type=dev, device_id=0,
                           element_type=_np_dtype(x.dtype), shape=tuple(x.shape),
                           buffer_ptr=x.data_ptr())
            odt = torch.float16 if self.onnx_out_fp16 else torch.float32
            out = torch.empty(self.batch, 3, self.H, self.W, dtype=odt,
                              device=x.device)
            iob.bind_output("y", device_type=dev, device_id=0,
                            element_type=_np_dtype(out.dtype), shape=tuple(out.shape),
                            buffer_ptr=out.data_ptr())
            self.session.run_with_iobinding(iob)
            return out
        import numpy as np                                    # noqa: PLC0415
        y = self.session.run(["y"], {"x": x.detach().cpu().numpy()})[0]
        return torch.from_numpy(y).to(self.device)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def interp_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """按 batch 上限自动切块处理任意数量。a/b: (N,3,H,W) 已补齐。"""
        outs = []
        for i in range(0, a.shape[0], self.batch):
            outs.append(self.infer(a[i:i + self.batch].contiguous(),
                                   b[i:i + self.batch].contiguous()))
        return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


# ---------------------------------------------------------------------- #
_ORT = None


def _ort():
    """延迟导入：必须先 import torch，让 torch 自带的 CUDA/cuDNN DLL 先进进程。

    本机 CUDA 是随 torch wheel 装在 `torch/lib` 的，不在系统 PATH 里；
    不先加载 torch，onnxruntime 的 CUDAExecutionProvider 会因为找不到
    cublasLt64 / cudnn 而静默退回 CPU。
    """
    global _ORT
    if _ORT is None:
        import os as _os
        lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
        if _os.path.isdir(lib):
            try:
                _os.add_dll_directory(lib)
            except OSError:
                pass
        import onnxruntime as ort                             # noqa: PLC0415
        _ORT = ort
    return _ORT


def _np_dtype(dtype: torch.dtype):
    import numpy as np                                        # noqa: PLC0415
    return np.float16 if dtype == torch.float16 else np.float32


# ---------------------------------------------------------------------- #
def export_rife_onnx(weight: str, out_path: str, size: tuple, fp16: bool = True,
                     opset: int = 18, verbose: bool = True):
    """把 RIFE 导出成固定 shape 的 ONNX。size=(H,W)，必须是 32 的倍数。"""
    h, w = size
    # 导出放在 CPU 上做：免得 dummy 输入和权重不在同一个设备上报错
    net = RifeEngine(weight=weight, backend="torch", fp16=False, batch=1,
                     use_cuda_graph=False, device="cpu", verbose=False)
    m = net.net.eval()                       # 已经是 _RifeExport
    # ★ 防御性：dummy 不要用全零 ★
    # BasicVSR++ 就因为全零 dummy 触发了 `is_mirror_extended` 判定，
    # 导致导出的图**缺了正向光流分支**、精度掉到 31 dB（见 build_trt_vsr_decomp.py）。
    # RIFE 本身没有数据相关分支（已核对），用非零 dummy 更保险。
    dummy = torch.rand(1, 6, h, w) * 0.5 + 0.25
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".fp32.tmp.onnx"
    # 必须用传统 TorchScript 导出器（dynamo=False）：
    # warplayer.warp 里有一个按 (device,size,dtype) 缓存的全局网格字典，
    # 新版 torch.export 不接受这种在 trace 期间被改写的全局状态，会直接报
    # "always_classified is unsupported"。
    torch.onnx.export(
        m, (dummy,), tmp, input_names=["x"], output_names=["y"],
        opset_version=opset, dynamo=False, do_constant_folding=True)
    if verbose:
        print(f"  [rife] 导出 fp32 ONNX -> {tmp}")
    if fp16:
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16
            import onnx                                        # noqa: PLC0415
            mdl = onnx.load(tmp)
            mdl16 = convert_float_to_float16(mdl, keep_io_types=True)
            onnx.save(mdl16, out_path)
            os.remove(tmp)
            if verbose:
                print(f"  [rife] 已转 fp16 -> {out_path}")
            return out_path
        except Exception as e:                                 # noqa: BLE001
            if verbose:
                print(f"  [rife] fp16 转换失败({e})，保留 fp32")
            os.replace(tmp, out_path)
            return out_path
    os.replace(tmp, out_path)
    return out_path
