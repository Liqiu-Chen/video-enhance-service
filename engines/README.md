# 引擎与权重

**这个目录里的文件不在 git 里**（体积超 GitHub 单文件 100 MB 限制），
请从本仓库的 **Releases** 页面下载后放到这里。

---

## 一、需要下载什么

| 文件 | 大小 | 用途 | 必需 |
|---|---|---|---|
| `vsr_tf32_t8_836x480_x4.trt` | 135 MB | **VSR 主引擎**（TensorRT TF32，t=8） | 推荐 |
| `vsr_tf32_t2_836x480_x4.trt` | 156 MB | VSR 引擎 t=2（小尺寸/快速验证） | 可选 |
| `RealESRGAN_x4plus.pth` | 64 MB | Real-ESRGAN 超分权重（备用后端） | 可选 |
| `basicvsr_plusplus_reds4.pth` | 28 MB | BasicVSR++ 权重（**重新构建引擎时需要**） | 构建时必需 |
| `spynet.pth` | 5.5 MB | SpyNet 光流权重（VSR 的对齐网络） | 构建时必需 |
| `flownet.pkl` | 11.6 MB | RIFE 插帧权重 | 必需 |

放好之后的目录结构：

```
video-enhance-service/
├── engines/
│   ├── vsr_tf32_t8_836x480_x4.trt
│   └── vsr_tf32_t2_836x480_x4.trt
├── weights/
│   ├── basicvsr_plusplus_reds4.pth
│   ├── spynet.pth
│   ├── RealESRGAN_x4plus.pth
│   └── flownet.pkl
└── ...
```

---

## 二、各 VSR 引擎的实测数据

对 PyTorch fp32 真值，真实视频帧 480×836 → ×4，两边都不分块：

| 引擎 | ms/帧 | 相对 PyTorch fp32 | PSNR | 最大差 | >2/255 | 推荐 |
|---|---|---|---|---|---|---|
| **`vsr_tf32_t*.trt`** | 260.9 | **1.69×** | **78.54 dB** | **1.0** | 0.000% | ⭐ **首选** |
| `*_fp32.trt`（已不在 Release） | 262.8 | 1.41× | 72.22 dB | 1.0 | 0.000% | 精度达标但更慢 |
| `*_fp16.trt`（已不在 Release） | 93.3 | 3.97× | 58.93 dB | 21.0 | 0.166% | 快，但**精度超标** |

**TF32 是"精度达标 + 有加速"的最优解**，所以只保留它。

> ⚠️ 这些引擎是**定形**的：`t8` 表示一次吃 8 帧、`836x480` 是输入尺寸。
> 换分辨率或换帧数需要重新构建（见下）。

---

## 三、重新构建引擎

### 0. 先装 TensorRT

```bash
pip install tensorrt pycuda onnx onnxruntime-gpu
```

### 1. 下载构建所需的权重

`basicvsr_plusplus_reds4.pth` 和 `spynet.pth` 放到 `weights/`。

### 2. 导出 ONNX

```bash
python tools/build/build_trt_vsr_decomp.py --size 836 480 --t 8 --scale 4 --mode fp32
```

**这个脚本会做三件事，顺序不能错：**
1. 打 `deform` 拆解补丁（否则导出直接报 `UnsupportedOperatorError`）
2. **用真实视频帧做 dummy**（用 `torch.zeros` 会导出残缺的图 —— 见主 README 的坑 2）
3. 导出并做常量折叠

导出后**务必自检**（这一步能省掉几小时）：
```bash
# 用 ONNX Runtime 跑一遍导出的图，和 PyTorch 对比
# 应该 > 90 dB；如果只有 50 多 dB，说明图是残缺的
python tools/build/export_vsr_ondemand.py --t 2 --size 836 480
```

### 3. 编译 TensorRT 引擎（TF32）

```bash
python - <<'PY'
import tensorrt as trt, os
D = "trt5"                    # ONNX 所在目录
os.chdir(D)
lg = trt.Logger(trt.Logger.WARNING); b = trt.Builder(lg)
nw = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
trt.OnnxParser(nw, lg).parse_from_file("vsr_decomp_t8_836x480_x4.onnx")
cfg = b.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30)
cfg.set_flag(trt.BuilderFlag.TF32)          # ★ 只开 TF32，不要设 FP16
ser = b.build_serialized_network(nw, cfg)
open("vsr_tf32_t8_836x480_x4.trt", "wb").write(ser)
PY
```

构建约 4~5 分钟（31009 层）。

**为什么只开 TF32 而不是 FP16**：
TF32 是 10 位尾数 + fp32 指数范围，对卷积的精度远好于 fp16，
速度又比纯 fp32 快。三者实测见上表。

### 4. 验证

```bash
python tools/bench/bench_trt_vsr.py --t 8 --size 836 480
```

应看到 PSNR ≈ 78 dB、最大差 ≈ 1/255。

---

## 四、RIFE 引擎（可选）

RIFE 的 TRT 引擎**精度不达标**（最大差 32/255），因此服务默认用
`rife_backend="torch"`（最大差 1/255）。如果需要极致速度再构建：

```bash
python tools/build/build_trt_rife.py --h 1696 --w 960
python tools/build/build_trt_rife_mixed.py --h 1696 --w 960   # 混合精度版，更准
```

**注意**：RIFE 的 `warp` 用 `padding_mode='border'`，而 TRT 只支持 zeros。
`flow_warp_trtsafe.py` 里有等价替换方案（坐标钳位），但 RIFE 本身没接这个补丁
（它的精度问题主要是 fp16 卷积，不是 padding）。
