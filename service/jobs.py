"""任务管理与工作区缓冲。

设计取舍
--------
* **单工作线程**：本机只有一块 GPU，而且 VSR ×4 在 5K 下就要 ~10 GiB 显存，
  并发跑两个任务只会互相抢显存然后双双 OOM。所以用「单 worker + 队列」，
  靠排队而不是靠并行来支撑多用户。
* **每个任务一个独立工作区** `data/jobs/<id>/`，任务之间零共享，
  删任务就是删目录。
* **配额**：受理前先看磁盘余量，不够直接拒；成品按 TTL 定期清理，
  避免"用户提交完就不管"把盘塞满。
"""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
JOBS = os.path.join(DATA, "jobs")

MAX_QUEUE = 16                  # 排队上限
JOB_TTL_H = 6.0                 # 成品保留小时数
MIN_FREE_GB = 8.0               # 低于这个磁盘余量直接拒收
MAX_UPLOAD_MB = 512


@dataclass
class Job:
    id: str
    src_name: str
    params: dict
    status: str = "queued"          # queued|running|done|failed|cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    _cancel: bool = False

    @property
    def workdir(self) -> str:
        return os.path.join(JOBS, self.id)

    @property
    def src_path(self) -> str:
        return os.path.join(self.workdir, "input" + os.path.splitext(self.src_name)[1])

    @property
    def out_path(self) -> str:
        return os.path.join(self.workdir, "output.mp4")

    def public(self) -> dict:
        d = asdict(self)
        d.pop("_cancel", None)
        d["has_result"] = os.path.exists(self.out_path)
        d["download_url"] = (f"/api/jobs/{self.id}/result"
                             if d["has_result"] and self.status == "done" else None)
        if self.started:
            d["queue_wait_s"] = round(self.started - self.created, 1)
        return d


class JobManager:
    def __init__(self, runner):
        """runner(job, progress) -> dict  实际执行推理的函数。"""
        self.runner = runner
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        self.lock = threading.Lock()
        os.makedirs(JOBS, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._janitor, daemon=True).start()

    # ------------------------------------------------------------------ #
    def submit(self, src_name: str, params: dict, src_file=None) -> Job:
        free = shutil.disk_usage(DATA).free / 2**30
        if free < MIN_FREE_GB:
            raise RuntimeError(f"磁盘余量不足（{free:.1f} GiB < {MIN_FREE_GB} GiB），"
                               f"请稍后再试或联系管理员清理")
        if self.q.qsize() >= MAX_QUEUE:
            raise RuntimeError(f"队列已满（{MAX_QUEUE}），请稍后再试")
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, src_name=os.path.basename(src_name), params=params)
        os.makedirs(os.path.join(job.workdir, "input"), exist_ok=True)
        if src_file is not None:
            dst = job.src_path
            with open(dst, "wb") as f:
                shutil.copyfileobj(src_file, f, 1 << 20)
        elif not os.path.exists(params.get("path", "")):
            raise RuntimeError("既没有上传文件，也没给可用的服务器路径")
        else:
            dst = job.src_path
            shutil.copy2(params["path"], dst)
        size = os.path.getsize(dst)
        if size == 0:
            shutil.rmtree(job.workdir, ignore_errors=True)
            raise RuntimeError("输入文件为空")
        with self.lock:
            self.jobs[jid] = job
        job.params["src_size_mb"] = round(size / 1048576, 1)
        self.q.put(jid)
        return job

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def list(self) -> list[dict]:
        with self.lock:
            js = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.public() for j in js]

    def cancel(self, jid: str, delete: bool = False) -> bool:
        j = self.jobs.get(jid)
        if j is None:
            return False
        if j.status in ("queued", "running"):
            j._cancel = True
            if j.status == "queued":
                j.status = "cancelled"
        if delete:
            with self.lock:
                self.jobs.pop(jid, None)
            shutil.rmtree(j.workdir, ignore_errors=True)
        return True

    # ------------------------------------------------------------------ #
    def _worker(self):
        while True:
            jid = self.q.get()
            j = self.jobs.get(jid)
            if j is None or j.status == "cancelled":
                continue
            j.status = "running"
            j.started = time.time()
            prog = {"stage": "start", "pct": 0.0}
            j.progress = prog
            try:
                res = self.runner(j, prog)
                if j._cancel:
                    j.status = "cancelled"
                else:
                    j.status = "done"
                    j.result = res
            except Exception as e:                             # noqa: BLE001
                j.status = "cancelled" if j._cancel else "failed"
                j.error = str(e)[:800]
                print("[job]", jid, "failed:", traceback.format_exc()[-1500:], flush=True)
            finally:
                j.finished = time.time()
                j.progress = dict(prog)

    def _janitor(self):
        """定期清理过期任务的工作区，避免把磁盘塞满。"""
        while True:
            time.sleep(600)
            now = time.time()
            for jid, j in list(self.jobs.items()):
                if j.status in ("done", "failed", "cancelled") and j.finished:
                    if now - j.finished > JOB_TTL_H * 3600:
                        with self.lock:
                            self.jobs.pop(jid, None)
                        shutil.rmtree(j.workdir, ignore_errors=True)
                        print(f"[janitor] 清理过期任务 {jid}", flush=True)
