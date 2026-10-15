"""推理服务：把 VSR ×4 → RIFE ×2 这条链路开放出去。

启动（默认监听 0.0.0.0:8000，局域网内其他机器可直接访问）：
    python -m uvicorn app:app --host 0.0.0.0 --port 8000

接口一览
--------
    GET  /                      网页上传界面
    GET  /api/health            服务与引擎状态
    POST /api/jobs              提交任务（multipart 上传 或 JSON 指定服务器路径）
    GET  /api/jobs              任务列表
    GET  /api/jobs/{id}         单个任务状态与进度
    GET  /api/jobs/{id}/result  下载结果视频
    DELETE /api/jobs/{id}       取消/删除任务
"""
from __future__ import annotations

import os
import secrets
import shutil
import sys
import threading
import time

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Query, Request,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "engine"))

from config import VSR_WEIGHT, RIFE_WEIGHT, RESR_WEIGHT, ensure_dirs  # noqa: E402
from jobs import MAX_UPLOAD_MB, JobManager  # noqa: E402
from pipeline import Pipeline, Progress  # noqa: E402

# ---------------------------------------------------------------------------
# 访问令牌
#
# 这个服务一旦开内网穿透就是公网可达的，而它**没有用户体系**。裸奔的后果：
#   1. 任何人提交任务都能白嫖你的 GPU；
#   2. `POST /api/jobs/json` 原本可以传服务器上的任意路径 —— 等于任意文件读取。
# 所以对外暴露前必须做两件事，都在这份代码里：
#   · 所有接口校验一个共享令牌（`INFER_TOKEN` 环境变量或启动时随机生成）；
#   · 本地路径提交被限制在 data/inbox 目录内，不能指向别处。
# ---------------------------------------------------------------------------
TOKEN = os.environ.get("INFER_TOKEN") or secrets.token_urlsafe(9)
INBOX = os.path.join(HERE, "data", "inbox")

app = FastAPI(title="VSR + RIFE 推理服务", version="1.1")

_engines: dict = {}
_engine_lock = threading.Lock()


def _unauth_html() -> str:
    return """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>需要访问令牌</title><style>body{background:#0f1115;color:#e6e8ec;
font:14px/1.7 "Microsoft YaHei",system-ui;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0}div{max-width:460px;padding:28px;
border:1px solid #2b3038;border-radius:10px;background:#171a21}
h1{font-size:17px;margin:0 0 12px;color:#5aa9ff}code{background:#1e2229;
padding:2px 7px;border-radius:4px;color:#8CD8FF}</style></head><body><div>
<h1>需要访问令牌</h1><p>这个推理服务只对持有令牌的访问者开放。</p>
<p>请在网址后面加上 <code>?token=你的令牌</code>，例如：</p>
<p><code>.../ ?token=xxxxxxxx</code></p>
<p style="color:#9aa3af">令牌由服务管理员提供。</p></div></body></html>"""


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    protected = path == "/" or path.startswith("/api")
    if protected:
        got = request.headers.get("x-api-token") or request.query_params.get("token")
        if not got or not secrets.compare_digest(got, TOKEN):
            if path == "/":
                return HTMLResponse(_unauth_html(), status_code=401)
            return JSONResponse(
                {"detail": "无效或缺失的访问令牌（用 X-API-Token 头或 ?token= 传入）"},
                status_code=401)
    return await call_next(request)


def get_engine(kind: str, **kw):
    """按需构建并缓存引擎（VSR 权重加载要几秒，不要每个任务都重建）。"""
    key = (kind,) + tuple(sorted(kw.items()))
    with _engine_lock:
        if key not in _engines:
            if kind == "vsr":
                from vsr_engine import VsrEngine
                _engines[key] = VsrEngine(weight=VSR_WEIGHT, **kw)
            elif kind == "vsr-trt":
                # TensorRT 版 BasicVSR++（可变形卷积已拆成 grid_sample+1x1conv
                # 原生算子）。引擎必须预先构建好，见 build_trt_vsr_decomp.py。
                from vsr_trt_engine import VsrTrtEngine
                _engines[key] = VsrTrtEngine(**kw)
            elif kind == "resr":
                from realesrgan_engine import RealEsrganEngine
                from config import RESR_WEIGHT
                _engines[key] = RealEsrganEngine(weight=RESR_WEIGHT, **kw)
            else:
                from rife_engine import RifeEngine
                _engines[key] = RifeEngine(weight=RIFE_WEIGHT, **kw)
    return _engines[key]


class RunParams(BaseModel):
    path: str | None = None            # 服务器本地路径（可选）
    interp: bool = True
    realtime: bool = True
    crf: int = 16
    preset: str = "medium"
    micro_batch: int = 12
    # 目标尺寸是 480p→1080p 这一档（720x1280 降 2 倍再 4x 超分 = 1440x2560）。
    # 实测这个尺寸下 BasicVSR++(chunk=8) 最快（172 ms/帧）且显存只要 3 GiB，
    # 还带时序一致性 —— 比单帧 Real-ESRGAN(222 ms/帧) 更好。
    # Real-ESRGAN 的优势是预热只要 0.5 s（VSR 要 8 s），短片段更划算。
    sr_backend: str = "vsr"            # vsr=BasicVSR++ | vsr-trt=TensorRT版(快4.4x) | resr=Real-ESRGAN
    vsr_chunk: int = 8                 # 目标尺寸下 chunk=8 最优（16/32 明显更慢）
    vsr_scale: int = 2                 # 2=720p→1440x2560（推荐）| 4=720p→2880x5120（超出实际需求）
    vsr_downsample: str = "linear"     # 2x 模式下用的下采样算法
    vsr_segments: int = 1              # 一次前向并行几段（>1 需实测有收益才用）
    resr_batch: int = 1                # 单帧 SR 一次前向吃几帧（实测 batch=1 最快）
    resr_tile: int = 0                 # >0 时按 tile×tile 分块（大分辨率才需要）
    fp16: bool = True
    # RIFE 实测最优：fp16 + fp32 warp（1.9x，PSNR 74 dB），CUDA Graph 主要用来压显存
    # （6.8 GiB -> 1.4 GiB），因为要和 VSR 的 ~10 GiB 共存
    rife_backend: str = "torch"        # torch(精度优先,最大差1/255) | trt(快2.3x但最大差32/255)        # torch | onnx（onnx 实测慢 3 倍，仅作对照）
    rife_batch: int = 1                # 实测 b1 比 b2 **更快**（288.4 vs 293.8 ms/对）
                                       # 且显存减半（3.39 vs 6.78 GiB）——
                                       # 5K 下 VSR 已占 ~10 GiB，这半个是救命的
    rife_graph: bool = False           # ⚠️ 默认关：Graph 会常驻一块显存，
                                       # 和 VSR 的 ~10 GiB 抢，5K 下会双双溢出到共享内存
    vsr_graph: bool = False            # VSR 分段大、显存本来就紧，默认不开


def _probe_source(src_path: str):
    """读出源视频的分辨率（自动配置要用）。"""
    from config import ffmpeg
    from pipeline import color_report
    m = color_report(ffmpeg(), src_path)
    return int(m["height"]), int(m["width"])


def _build_sr_engine(h0: int, w0: int, out_scale: int, fp16: bool = True,
                     force_backend: str = "", verbose: bool = True):
    """按输出分辨率**自动**挑超分后端与并行度，并用真实峰值显存验证。

    候选链来自 `autoconfig`（实测数据拟合）。逐个试：建引擎 → 跑一次预热 →
    看 `max_memory_allocated` 是否在预算内，超了就换下一个候选。
    静态估算不够准（桌面程序也占显存），所以以实测为准。
    """
    from autoconfig import candidates, probe_ok, VRAM_BUDGET_GB
    last_err = None
    for be, params in candidates(h0, w0, out_scale, backend=force_backend or None):
        try:
            if be == "vsr":
                eng = get_engine("vsr", variant="basicvsr_plusplus", fp16=fp16,
                                 overlap=2, out_scale=out_scale,
                                 downsample="linear", segments=1,
                                 chunk=int(params.get("chunk", 8)),
                                 cudnn_benchmark=False)
            else:
                eng = get_engine("resr", fp16=fp16,
                                 batch=int(params.get("batch", 1)),
                                 tile=int(params.get("tile", 0)),
                                 out_scale=out_scale, downsample="linear")
            eng.warmup(h0, w0)
            ok, peak = probe_ok(eng, h0, w0)
            if ok:
                if verbose:
                    print(f"  [auto] 选用 {be} {params}  实测峰值 {peak:.2f} GiB "
                          f"(预算 {VRAM_BUDGET_GB:.0f})", flush=True)
                return eng, be, params
            if verbose:
                print(f"  [auto] {be} {params} 峰值 {peak:.2f} GiB 超预算，降级",
                      flush=True)
        except Exception as e:                                 # noqa: BLE001
            last_err = e
            if verbose:
                print(f"  [auto] {be} {params} 失败：{type(e).__name__}: "
                      f"{str(e)[:80]}，降级", flush=True)
            torch.cuda.empty_cache()
    raise RuntimeError(f"所有超分配置都跑不起来，最后一个错误：{last_err}")


def _runner(job, prog: dict):
    """任务执行体：在 worker 线程里跑完整流水线。"""
    p = dict(job.params)
    t0 = time.time()

    def on_prog(stage, done, total, msg=""):
        prog.update({"stage": stage, "done": done, "total": total,
                     "pct": round(done / total * 100, 1) if total else 0.0,
                     "message": msg,
                     "elapsed_s": round(time.time() - t0, 1)})

    # 超分后端：默认**自动**按分辨率挑（见 autoconfig.py）；
    # 只有用户显式勾了"手动指定后端"才按给定参数来。
    out_scale = int(p.get("vsr_scale", 2))
    h0, w0 = _probe_source(job.src_path)
    if p.get("sr_backend_explicit", False):
        if p.get("sr_backend") == "resr":
            vsr = get_engine("resr", fp16=p.get("fp16", True),
                             batch=int(p.get("resr_batch", 1)),
                             tile=int(p.get("resr_tile", 0)),
                             out_scale=out_scale)
        else:
            vsr = get_engine("vsr", variant="basicvsr_plusplus",
                             fp16=p.get("fp16", True),
                             chunk=int(p.get("vsr_chunk", 8)), overlap=2,
                             out_scale=out_scale,
                             downsample=p.get("vsr_downsample", "linear"),
                             segments=int(p.get("vsr_segments", 1)),
                             cudnn_benchmark=False)
        vsr.warmup(h0, w0)
    else:
        from autoconfig import describe
        print(f"  [auto] {describe(h0, w0, out_scale)}", flush=True)
        vsr, _be, _bp = _build_sr_engine(h0, w0, out_scale,
                                         fp16=p.get("fp16", True))
        p["sr_backend"] = _be
        p["auto_choice"] = {**_bp, "backend": _be}
    p["src_hw"] = [h0, w0]
    rife = None
    if p.get("interp", True):
        # ★ 默认回到 torch ★
        # 实测（真实连续帧、对 fp32 真值）：
        #   torch fp16 : 最大差 1.0/255，50% 像素与 fp32 逐位相同 —— 达标
        #   TRT        : 最大差 32/255，>2/255 占 0.30%，最大连通块 2145 像素
        # TRT 虽快 2.3x，但有可见的块状伪影，不满足精度要求，故不作默认。
        # 需要速度时显式传 rife_backend="trt"。
        want = p.get("rife_backend", "torch")
        kw = dict(fp16=p.get("fp16", True), batch=int(p.get("rife_batch", 1)),
                  use_cuda_graph=bool(p.get("rife_graph", True)))
        try:
            rife = get_engine("rife", backend=want, **kw)
        except Exception as e:                                 # noqa: BLE001
            print(f"  [rife] backend={want} 不可用（{type(e).__name__}: "
                  f"{str(e)[:80]}），退回 torch", flush=True)
            rife = get_engine("rife", backend="torch", **kw)

    pipe = Pipeline(vsr, rife, micro_batch=int(p.get("micro_batch", 12)), verbose=True)
    pr = Progress()
    pr.total = 0

    # 把 Progress 的实时值同步到对外可见的 prog
    stop_flag = {"v": False}
    sync = threading.Thread(target=_sync_progress, args=(pr, prog, stop_flag), daemon=True)
    sync.start()
    try:
        res = pipe.run(job.src_path, job.out_path,
                       interp=p.get("interp", True),
                       realtime=p.get("realtime", True),
                       crf=int(p.get("crf", 16)),
                       preset=p.get("preset", "medium"),
                       progress=pr,
                       should_stop=lambda: job._cancel)
    finally:
        stop_flag["v"] = True
        sync.join(timeout=2)
    return res


def _sync_progress(pr: Progress, prog: dict, stop: dict):
    while not stop["v"]:
        d = pr.as_dict()
        prog.update(d)
        time.sleep(0.5)


def _resolve_inbox(path: str) -> str:
    """把用户给的服务器路径限制在 data/inbox 目录内。

    原来 `POST /api/jobs/json` 可以传**任意**路径，服务一暴露就等于
    "任意文件读取 + 拿去推理"。现在只允许 inbox 里的文件，
    想走本地路径就先把它放进 `推理服务/data/inbox/`。
    """
    os.makedirs(INBOX, exist_ok=True)
    cand = os.path.abspath(os.path.join(INBOX, os.path.basename(path)))
    if not os.path.isfile(cand):
        raise HTTPException(
            400, f"只接受 data/inbox 目录内的文件。请把视频放到 {INBOX} 后，"
                 f"用 path=文件名 提交（其中可用的文件："
                 f"{sorted(os.listdir(INBOX))[:20]}）")
    return cand


def _parse_form(form: dict) -> dict:
    def b(k, d=False):
        v = form.get(k)
        if v is None:
            return d
        return str(v).lower() in ("1", "true", "yes", "on")

    def i(k, d):
        try:
            return int(form.get(k, d))
        except (TypeError, ValueError):
            return d

    return {
        "sr_backend": form.get("sr_backend", "vsr"),
        "vsr_scale": i("vsr_scale", 2),
        "resr_batch": i("resr_batch", 1),
        "resr_tile": i("resr_tile", 0),
        "interp": b("interp", True),
        "realtime": b("realtime", True),
        "crf": i("crf", 16),
        "preset": form.get("preset", "medium"),
        "micro_batch": i("micro_batch", 12),
        "vsr_chunk": i("vsr_chunk", 8),
        "fp16": b("fp16", True),
        "rife_backend": form.get("rife_backend", "torch"),
        "rife_batch": i("rife_batch", 1),
        "cuda_graph": b("cuda_graph", False),
    }


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(default=None),
                     path: str | None = Form(default=None),
                     interp: str = Form("true"),
                     realtime: str = Form("true"),
                     crf: int = Form(16),
                     preset: str = Form("medium"),
                     micro_batch: int = Form(12),
                     vsr_chunk: int = Form(8),
                     fp16: str = Form("true"),
                     rife_backend: str = Form("torch"),
                     rife_batch: int = Form(1),
                     rife_graph: str = Form("true"),
                     vsr_graph: str = Form("false"),
                     sr_backend: str = Form("vsr"),
                     vsr_scale: int = Form(2),
                     resr_batch: int = Form(1),
                     resr_tile: int = Form(0)):
    params = _parse_form(dict(interp=interp, realtime=realtime, crf=crf, preset=preset,
                              micro_batch=micro_batch, vsr_chunk=vsr_chunk, fp16=fp16,
                              rife_backend=rife_backend, rife_batch=rife_batch,
                              rife_graph=rife_graph, vsr_graph=vsr_graph,
                              sr_backend=sr_backend, vsr_scale=vsr_scale,
                              resr_batch=resr_batch, resr_tile=resr_tile))
    if file is None and not path:
        raise HTTPException(400, "需要上传文件，或用 path 指定 data/inbox 里的视频")
    if file is not None and file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_MB} MB 上限")
    if file is None:
        path = _resolve_inbox(path)
    name = file.filename if file is not None else os.path.basename(path)
    try:
        job = MANAGER.submit(name, params,
                             src_file=file.file if file is not None else None)
    except Exception as e:                                     # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    return JSONResponse(job.public(), status_code=202)


@app.post("/api/jobs/json")
async def create_job_json(p: RunParams):
    if not p.path:
        raise HTTPException(400, "需要 path，指向 data/inbox 里的视频")
    payload = p.model_dump()
    payload["path"] = _resolve_inbox(p.path)
    try:
        job = MANAGER.submit(os.path.basename(payload["path"]), payload)
    except Exception as e:                                     # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": MANAGER.list()}


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    j = MANAGER.get(jid)
    if j is None:
        raise HTTPException(404, "任务不存在")
    return j.public()


@app.delete("/api/jobs/{jid}")
async def delete_job(jid: str, delete: bool = True):
    if not MANAGER.cancel(jid, delete=delete):
        raise HTTPException(404, "任务不存在")
    return {"ok": True}


@app.get("/api/jobs/{jid}/result")
async def get_result(jid: str):
    j = MANAGER.get(jid)
    if j is None:
        raise HTTPException(404, "任务不存在")
    if not os.path.exists(j.out_path):
        raise HTTPException(409, f"结果尚未就绪（当前状态 {j.status}）")
    return FileResponse(j.out_path, media_type="video/mp4",
                        filename=f"{os.path.splitext(j.src_name)[0]}_vsr4x_fi2x.mp4")


@app.get("/api/health")
async def health():
    import torch
    free = shutil.disk_usage(os.path.dirname(HERE)).free / 2**30
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else 0,
        "vram_used_gb": round(torch.cuda.memory_reserved() / 2**30, 2)
        if torch.cuda.is_available() else 0,
        "disk_free_gb": round(free, 1),
        "queued": MANAGER.q.qsize(),
        "engines_loaded": len(_engines),
        "defaults": RunParams().model_dump(),
    }


@app.get("/api/tunnel")
async def tunnel_info():
    """返回隧道进程写下的**当前**公网地址。

    免费隧道断线重连会换域名，用户手里的旧链接就失效了（表现为
    页面/提交报错）。局域网地址永不改变，所以从这里能拿到最新的公网地址。
    """
    f = os.path.join(HERE, "current_url.txt")
    url = ""
    try:
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                url = fh.read().strip()
    except OSError:
        pass
    return {"tunnel_url": url, "has_tunnel": bool(url), "token": TOKEN}


@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(HERE, "static", "index.html")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>VSR + RIFE 推理服务</h1><p>见 /docs</p>")


ensure_dirs()
MANAGER = JobManager(_runner)


def _startup_warmup():
    """服务启动时把 VSR 引擎建好并预热。

    **为什么必须做**：cuDNN/cuBLAS 对每个新 shape 的首次调用要付一大笔初始化
    （实测 16 帧任务里"预热"阶段独占 **70.7 s**，占整个任务 40%）。
    引擎在服务里是常驻的，所以这笔钱只该付一次 —— 放在启动阶段，
    而不是让第一个提交任务的用户莫名其妙等 70 秒。

    放后台线程跑，不挡住端口监听。
    """
    import torch
    t0 = time.time()
    try:
        vsr = get_engine("vsr", variant="basicvsr_plusplus", fp16=True, chunk=8,
                         overlap=2, out_scale=2, downsample="linear", segments=1)
        vsr.warmup(720, 1280)
        torch.cuda.empty_cache()
        print(f"[warmup] VSR 引擎已预热（{time.time()-t0:.1f}s）", flush=True)
    except Exception as e:                                     # noqa: BLE001
        print(f"[warmup] 失败，不影响使用: {e}", flush=True)


threading.Thread(target=_startup_warmup, daemon=True).start()
