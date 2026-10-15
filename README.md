# video-enhance-service

**视频超分（VSR）×4 + 插帧（RIFE）×2 的 TensorRT 加速推理服务。**

带一个 FastAPI 网页端（上传 → 异步任务 → 进度 → 下载），
以及从 PyTorch 模型到 TensorRT 引擎的完整构建链路。

> 这套代码是在 RTX 5070 Ti（Blackwell sm_120 / 16 GB）上开发并实测的。
> 硬件不同时，性能数字会变，但**精度结论和踩过的坑是通用的**。

---

## 一、目录结构

```
video-enhance-service/
├── README.md
├── requirements.txt
├── .gitignore
│
├── service/                       ★ 服务本体（可直接跑）
│   ├── app.py                     FastAPI 应用：REST API + 静态页
│   ├── serve.py                   启动入口（读参数 → uvicorn）
│   ├── pipeline.py                ★ 核心流水线：VSR → RIFE 串联 + 有界缓冲 + 微批
│   ├── jobs.py                    异步任务管理（提交/进度/取消/产物）
│   ├── config.py                  路径与权重配置（含 vendor 回退）
│   ├── autoconfig.py              ★ 按输出分辨率自动选后端与参数（防爆显存）
│   ├── public.py                  公网隧道（localhost.run）
│   ├── mp4meta.py                 解析 MP4 colr box 取色彩元数据（环境无 ffprobe）
│   ├── engine/                    推理后端（统一接口，可互换）
│   │   ├── vsr_engine.py          BasicVSR++（PyTorch，支持 autocast 混合精度）
│   │   ├── realesrgan_engine.py   Real-ESRGAN（逐帧独立，显存友好）
│   │   ├── rife_engine.py         RIFE（torch / onnx / trt 三条后端）
│   │   └── vsr_trt_engine.py      ★ VSR 的 TensorRT 适配器（默认 TF32 引擎）
│   └── static/index.html          网页端界面
│
├── tools/
│   ├── build/                     ★ 引擎构建链路（从模型 → ONNX → TRT）
│   │   ├── deform_decompose.py     可变形卷积拆解（TRT 不支持 deform_conv 的绕法）
│   │   ├── flow_warp_trtsafe.py    border padding 的等价替换
│   │   ├── build_trt_vsr_decomp.py ★ VSR 引擎构建（含"真实帧 dummy"的关键修复）
│   │   ├── build_trt_rife.py       RIFE 引擎构建
│   │   ├── build_trt_rife_mixed.py RIFE 混合精度（GridSample/Resize 留 fp32）
│   │   ├── build_trt_vsr_mixed.py  VSR 混合精度（同类思路）
│   │   ├── build_trt_vsr_precise.py / build_trt_vsr_trtsafe.py / build_trt_vsr.py
│   │   │                            VSR 精度攻坚期间的几个变体（保留作参考）
│   │   ├── export_vsr_ondemand.py  VSR ONNX 导出（真实帧 dummy 的最小复现）
│   │   └── export_onnx.py          RIFE ONNX 导出
│   │
│   ├── bench/                     性能基准
│   │   ├── bench_vsr.py / bench_rife.py / bench_resr.py     单模型
│   │   ├── bench_pipe_vs_iso.py   流水线 vs 隔离（帧数必须对齐，否则结论是假的）
│   │   ├── bench_e2e.py / bench_target.py / bench_interleave.py
│   │   ├── bench_trt_vsr.py       VSR 引擎精度/速度对照
│   │   ├── bench_cudnn_careful.py cudnn.benchmark 的陷阱测量
│   │   ├── profile_vsr.py         分段耗时剖析
│   │   └── exp_vsr.py             探索性实验
│   │
│   └── diag/                      诊断 / 验证
│       ├── diag_error_map.py      ★ 误差的空间分布（边框 vs 内部、连通域大小）
│       ├── diag_vsr_precision.py  VSR 逐帧误差 + ONNX Runtime 对照
│       ├── diag_vsr_fp16.py       ★ 定位"fp16 用错了方式"的诊断
│       ├── diag_layer_bisect.py   逐层二分找发散点
│       ├── diag_unhalf_bisect.py  逐个模块解除 fp16，找敏感模块
│       ├── diag_trt_rife.py / diag_fp16.py
│       ├── verify_rife_engines.py / verify_rife_real.py
│       ├── vsr_decompose_test.py  ★ 拆解 vs torchvision 的数值等价自测
│       ├── test_engine_warp.py
│       └── smoke_pipeline.py      流水线冒烟测试
│
├── vendor/                        模型源码（第三方，MIT，随仓库分发以便自包含）
│   ├── vsr/vsr/                   BasicVSR++ / SPyNet（OpenMMLab，含本地性能修正）
│   ├── rife/rife_ms/              RIFE（Megvii）
│   └── realesrgan/realesrgan/     Real-ESRGAN（Xintao Wang）
│
└── engines/                       引擎与权重（体积大，走 GitHub Release）
    └── README.md                  下载说明 + 各引擎的实测精度/速度表
```

**所有可执行脚本都自带路径引导**，在仓库任意位置直接 `python tools/xxx/yyy.py` 即可，
不需要设置 `PYTHONPATH`，也不需要 `cd` 到特定目录。

---

## 二、快速开始

### 1. 环境

```bash
pip install -r requirements.txt
```

**显卡是 Blackwell（sm_120，RTX 50 系）时必须用 cu128 及以上的轮子**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
用 cu121 及更早的轮子会在 sm_120 上直接报 `no kernel image is available`。

### 2. 权重与引擎

```bash
# 看 engines/README.md，从 Release 下载
```

### 3. 跑起来

```bash
python service/serve.py --host 0.0.0.0 --port 8000
# 或
python -m uvicorn app:app --host 0.0.0.0 --port 8000   # 需先 cd service/
```

浏览器打开 `http://127.0.0.1:8000`。

命令行批量处理：
```bash
python tools/diag/smoke_pipeline.py --clip 输入.mp4 --out 输出.mp4
```

---

## 三、实测性能与精度

**环境**：RTX 5070 Ti（Blackwell sm_120，16 GB）+ CUDA 12.8 + TensorRT 10.x。
**素材**：480×836 → ×4 → 1920×3344 的真实视频帧，两边都不分块、同帧数。

### VSR（BasicVSR++）—— 对 PyTorch fp32 真值

| 实现 | ms/帧 | 相对 PyTorch fp32 | PSNR | **最大差** | >2/255 |
|---|---|---|---|---|---|
| PyTorch fp32（基准） | 441.9 | 1.00× | — | 0 | 0% |
| PyTorch fp16（**autocast**） | — | — | **99.00 dB** | **0.0** | 0% |
| **TensorRT TF32** ⭐ | **260.9** | **1.69×** | **78.54 dB** | **1.0** | **0.000%** |
| TensorRT fp32 | 262.8 | 1.41× | 72.22 dB | 1.0 | 0.000% |
| TensorRT fp16 | 93.3 | 3.97× | 58.93 dB | 21.0 | 0.166% |

**推荐 TF32**：精度最高（比 fp32 引擎还高）、且比 PyTorch fp32 快 1.69×。

### 三个必须知道的坑（都是实测踩出来的）

#### 坑 1：fp16 不能用 `net.half()`，要用 `autocast`

```python
net = net.half()                                    # ❌ 30.11 dB / 最大差 102
with torch.autocast("cuda", dtype=torch.float16):   # ✅ 66.98 dB / 最大差 1
    out = net(x)
```

`net.half()` 把**整个网络连权重带激活**粗暴转 fp16，
而 BasicVSR++ 里有对数值范围敏感的部分（SpyNet 光流的点积相关、时序递归累加、
可变形卷积的坐标与掩码），整网 fp16 会把它们推到精度边缘。
`autocast` 让卷积走 fp16、敏感算子自动留 fp32，**用 0.27× 的速度换回 36.7 dB 精度**。

#### 坑 2（最贵的一个）：导出的 dummy 绝不能用 `torch.zeros`

`BasicVSRPlusPlusNet` 里有个**实例标志** `is_mirror_extended`：

```python
if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:   # ← 全零输入时恒成立！
    self.is_mirror_extended = True            # ← 只置 True，永不重置
...
if self.is_mirror_extended:
    flows_forward = None                      # ← 直接跳过正向光流计算！
```

用 `torch.zeros` 做 dummy 导出 → 标志被置 True → **导出的图缺少正向光流分支**，
是个结构残缺的网络。实测（ORT vs PyTorch）：

| dummy | `is_mirror_extended` | ORT vs PyTorch |
|---|---|---|
| `torch.zeros` | True | **55.89 dB** / 最大差 32 ❌ |
| **真实帧** | False | **98.64 dB** / 最大差 1 ✅ |

这个坑把 TRT 精度一直卡在 **31 dB**，而在残缺图上做的所有"精度手术"
（GridSample fp32 / Resize fp32 / 499 个卷积 fp32 / border→zeros）**全部无效**。
换成真实帧 dummy 后，精度直接跳到 58.93 dB（fp16）/ 78.54 dB（TF32）。

**血泪教训：导出/编译前先想想 dummy 会不会触发模型里的数据相关分支。**

#### 坑 3：TRT 不支持 `deform_conv2d`，也不支持 GridSample 的 border padding

- `torchvision.ops.deform_conv2d` 是自定义算子，ONNX 到 opset 28 都没有带 mask 的
  DeformConv。解法是 `deform_decompose.py` 把它**拆解**成
  9 次 `grid_sample` + mask 相乘 + 1×1 卷积，数值等价性有自测保证
  （相对误差 ~3e-6，`python tools/diag/vsr_decompose_test.py`）。
- TRT 的 GridSample **只支持 zeros padding**，ONNX 里写 border 会被静默忽略。
  解法在 `flow_warp_trtsafe.py`：`align_corners=True` 下把归一化坐标钳到 `[-1,1]`
  就**严格等价于** border（实测最大差 0.000e+00）。

---

## 四、这个仓库的代码风格约定

1. **注释只写"为什么"，不写"是什么"**。凡是反直觉的写法、踩过的坑、
   被否掉的方案，都在注释里留了实测数字 —— 目的只有一个：**别再踩第二次**。
2. **性能结论必须绑定尺寸**。同一份代码在 2880×5120 和 1440×2560 上
   最优配置可能是相反的（例如 chunk 该取几、要不要合批）。
3. **比性能必须统一帧数**。拿"240 帧的隔离跑"对比"整段素材的流水线跑"
   会得出"流水线慢 4.88 倍"的假结论。
4. **比精度必须统一分块**。BasicVSR++ 是**双向传播**，补帧会改变**全部**帧
   而非只影响末尾；分块本身就能造成 51~55 dB 量级的差异。
5. **看精度不能只看最大差**，要同时看中位数/分位数、
   **超阈像素占比**、以及**最大连通域大小**
   （散点肉眼不可见，成片才是伪影）。

---

## 五、许可

- 本项目自身代码：**MIT**
- `vendor/` 下为第三方代码，版权归原作者：
  - **BasicVSR++ / SPyNet** — OpenMMLab，Apache-2.0
  - **RIFE** — Megvii，MIT
  - **Real-ESRGAN** — Xintao Wang，BSD-3-Clause
