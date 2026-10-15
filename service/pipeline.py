"""VSR → RIFE 串联推理流水线（带缓冲管理）。

数据流
------
    源视频 ──ffmpeg解码──▶ [入队缓冲] ──▶ VSR ×4 ──▶ RIFE ×2 ──▶ [出队缓冲] ──ffmpeg编码──▶ 成品

缓冲管理的三条原则
------------------
1. **中间结果全部在内存里传递，不落盘。** 原方案把每帧写成 PNG（2880x5120 约 20 MB）
   再由下一级读回来，实测 PNG 编码 0.62 s/帧 + 解码 0.2 s/帧，比模型本身还慢。
   这里改成有界队列直传。
2. **每个队列都有上限**，拉满就阻塞上游 —— 内存占用有硬上限，不会因为
   上下游速度不匹配而无限堆积（之前在内存里攒 791 帧直接 OOM 过）。
3. **微批处理**：把源帧切成 `micro_batch` 大小的块依次过两级模型，
   块内在两模型之间也不落盘。这样峰值内存 ≈ 几块帧，与视频总长无关。

线程模型
--------
    读取线程(ffmpeg 解码) ──▶ queue_in ──▶ 主线程(VSR+RIFE) ──▶ queue_out ──▶ 写线程(ffmpeg 编码)
解码/编码是 CPU + IO 密集，推理是 GPU 密集，三者重叠可以把总时间压到
`max(IO, GPU)` 而不是相加。
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from config import ffmpeg, BASE  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "engine") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "engine"))
# 复用已有的 MP4 色彩元数据解析（colr box 解析 + ffmpeg 参数生成）
_MP4META_DIR = os.path.join(BASE, "超分模型", "VSR", "tools")
if _MP4META_DIR not in sys.path:
    sys.path.insert(0, _MP4META_DIR)

from mp4meta import (color_report, ffmpeg_args, setparams_filter,  # noqa: E402
                     true_fps)


class Progress:
    def __init__(self):
        self.stage = "init"
        self.done = 0
        self.total = 0
        self.message = ""
        self.t0 = time.time()
        self.marks: dict[str, float] = {}

    def as_dict(self, elapsed=None):
        el = elapsed if elapsed is not None else time.time() - self.t0
        return {
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "pct": round(self.done / self.total * 100, 1) if self.total else 0.0,
            "message": self.message,
            "elapsed_s": round(el, 1),
            "fps": round(self.done / el, 3) if el > 0 and self.done else 0.0,
            "stages": self.marks,
        }


def _count_frames(ff: str, path: str):
    p = subprocess.run([ff, "-hide_banner", "-i", path, "-map", "0:v:0",
                        "-c", "copy", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    import re
    m = re.findall(r"frame=\s*(\d+)", p.stderr)
    return int(m[-1]) if m else None


class Pipeline:
    """一次推理任务的执行体。可被服务端在子线程里调用。"""

    def __init__(self, vsr_engine, rife_engine=None, *, micro_batch: int = 12,
                 queue_in: int = 6, queue_out: int = 24,
                 cache_flush_gb: float = 8.0, verbose: bool = True):
        self.vsr = vsr_engine
        self.rife = rife_engine
        self.micro_batch = int(micro_batch)
        self.qin_size = int(queue_in)
        self.qout_size = int(queue_out)
        # 预留显存超过这个值才主动 empty_cache。目标尺寸（1440x2560）下 VSR 只占
        # ~3 GiB，不会触发；只有回到 2880x5120 那种高占用场景才会触发。
        self.cache_flush_gb = float(cache_flush_gb)
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print("  [pipe]", *a, flush=True)

    @staticmethod
    def _has_audio(ff: str, path: str) -> bool:
        """源视频有没有音轨。有的话要在最后合并回成品里。"""
        r = subprocess.run([ff, "-hide_banner", "-i", path],
                           capture_output=True, text=True, errors="replace")
        return "Audio:" in r.stderr

    @staticmethod
    def _probe_ok(path: str) -> bool:
        """产物能不能被正常解析。

        编码器没写完（缺 moov atom）时文件照样有几百 MB，但根本打不开 ——
        光看大小判断不出来，必须让 ffmpeg 真的读一次。
        """
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            return False
        p = subprocess.run([ffmpeg(), "-hide_banner", "-i", path],
                           capture_output=True, text=True, errors="replace")
        return "Duration:" in p.stderr and "Video:" in p.stderr

    # ------------------------------------------------------------------ #
    def run(self, src: str, out: str, *, interp: bool = True,
            realtime: bool = True, crf: int = 16, preset: str = "medium",
            progress: Progress | None = None,
            should_stop=None) -> dict:
        ff = ffmpeg()
        pg = progress or Progress()
        meta = color_report(ff, src)
        fps_src = float(true_fps(ff, src) or meta.get("fps") or 25.0)
        W0, H0 = int(meta["width"]), int(meta["height"])
        W0 -= W0 % 2
        H0 -= H0 % 2
        n_src = _count_frames(ff, src) or int(round(meta.get("duration", 0) * fps_src))
        factor = 2 if (interp and self.rife is not None) else 1
        fps_out = fps_src * (factor if realtime else 1)
        n_out_expect = n_src * factor
        dur_src = n_src / fps_src
        out_dur = n_out_expect / fps_out            # realtime -> == dur_src

        W1, H1 = W0 * self.vsr.scale, H0 * self.vsr.scale
        _t = time.time()
        def _phase(tag):
            nonlocal _t
            now = time.time()
            self._log(f"[阶段] {tag:<22}{now-_t:7.1f}s")
            _t = now

        pg.total = n_src
        pg.stage = "decode"
        pg.message = (f"{W0}x{H0}@{fps_src:g}fps {n_src}帧 -> "
                      f"{W1}x{H1}@{fps_out:g}fps {n_out_expect}帧")

        qin: queue.Queue = queue.Queue(maxsize=self.qin_size)
        qout: queue.Queue = queue.Queue(maxsize=self.qout_size)
        errors: list = []
        # 编码器/解码器一旦死掉就置位，主循环据此及时退出，
        # 避免"消费者没了、生产者还往满了的队列里塞"而永久阻塞（见 _put）。
        dead = threading.Event()

        def _put(q, item, timeout=1.0):
            """带超时的入队。

            「队列满 + 消费者已经退出」是死锁的经典组合：生产者会永远卡在 put 上。
            之前就栽在这里 —— 编码器提前退出后 writer 线程跟着结束，主循环还在往
            满队列里塞帧，于是永久阻塞；而进度计数器**在 put 之前**就已经置成 100%，
            所以页面上显示"已完成"，实际状态还是 running，也就没有下载链接。
            这里超时后检查 dead 标志，及时抛错而不是无限等。
            """
            while True:
                try:
                    q.put(item, timeout=timeout)
                    return
                except queue.Full:
                    if dead.is_set():
                        raise RuntimeError(
                            "下游已停止（编码器/解码器异常退出），流水线中止")

        # ---- 解码线程 ----
        dec = subprocess.Popen(
            [ff, "-v", "error", "-i", src, "-fps_mode", "passthrough",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-vf", f"crop={W0}:{H0}:0:0", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 22)

        def drain(proc, tag):
            """必须把 stderr 读干净。

            ffmpeg 的 stderr 是管道，缓冲区（约 64 KB）写满后会**阻塞 ffmpeg 自己**，
            没人读就会死锁。同时也只有读了才能把出错原因报出来。
            """
            try:
                for line in iter(proc.stderr.readline, b""):
                    t = line.decode("utf-8", "replace").strip()
                    if t:
                        errors.append(f"{tag}: {t}")
                        self._log(f"[{tag}] {t}")
            except Exception:                                  # noqa: BLE001
                pass

        def reader():
            fsz = W0 * H0 * 3
            try:
                while not dead.is_set():
                    buf = dec.stdout.read(fsz)
                    if len(buf) < fsz:
                        break
                    _put(qin, np.frombuffer(buf, np.uint8).reshape(H0, W0, 3))
            except Exception as e:                             # noqa: BLE001
                errors.append(f"decode: {e}")
            finally:
                # 投递 EOF 哨兵 **必须可靠**。
                #
                # ⚠️ 这里踩过一个很难查的坑：原来写成
                #     try: qin.put_nowait(None)
                #     except queue.Full: pass
                # —— 队列满时干脆把哨兵丢掉，主循环的 qin.get() 就永远等下去
                # （栈上表现为：主线程卡 qin.get、writer 卡 qout.get、reader 已退出）。
                # micro_batch 刚好整除帧数时最容易凑出"队列满"的时序。
                #
                # 但也不能简单改回阻塞 put(None)：那会在消费者已经死掉时永久卡住
                # （上一轮的死锁就是这么来的）。所以用「带超时的阻塞 put + dead 检查」，
                # 两种情况都能正确处理。
                while not dead.is_set():
                    try:
                        qin.put(None, timeout=0.5)
                        break
                    except queue.Full:
                        continue

        # ---- 编码线程 ----
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        enc_cmd = [ff, "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{W1}x{H1}", "-framerate", f"{fps_out}", "-i", "-"]
        sp = setparams_filter(meta)
        if sp:
            enc_cmd += ["-vf", sp]
        enc_cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-pix_fmt", meta.get("pix_fmt", "yuv420p")]
        enc_cmd += ffmpeg_args(meta)
        # 这里**不加 -t**：新流水线不挂音轨（下面没有 -map 音频），
        # 不存在"音轨比视频长把成品撑长"的问题，写入多少帧就是多少帧。
        # 老实现里那个 -t 是按「秒」截断的，边界上可能让 ffmpeg 提前停止读取输入，
        # 之后 writer 的 write 就会拿到 BrokenPipe —— 这正是本次卡死的触发点。
        # 保险起见用 -frames:v 按**帧数**封顶（精确、且不会提前截断）。
        enc_cmd += ["-frames:v", str(n_out_expect)]
        # 编码器先只出**纯视频**，音轨在后面单独合并。
        # 原因：这里是 rawvideo 管道输入，本来就没有音频流可 map；
        # 而用户的源视频往往带音轨，直接输出会静默丢音。
        vid_only = out + ".video.mp4"
        enc_cmd += ["-movflags", "+faststart", vid_only]
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=1 << 24)
        written = [0]

        def writer():
            try:
                while True:
                    fr = qout.get()
                    if fr is None:
                        break
                    enc.stdin.write(fr.tobytes())
                    written[0] += 1
            except Exception as e:                             # noqa: BLE001
                # 编码器一旦提前退出，这里的 write 会 BrokenPipe；必须立刻置 dead，
                # 让主循环停下来，否则就是上面说的那个死锁。
                errors.append(f"encode: {type(e).__name__}: {e}")
                dead.set()
            finally:
                try:
                    enc.stdin.close()
                except Exception:                              # noqa: BLE001
                    pass

        tr = threading.Thread(target=reader, daemon=True)
        tw = threading.Thread(target=writer, daemon=True)
        td1 = threading.Thread(target=drain, args=(dec, "decode-stderr"), daemon=True)
        td2 = threading.Thread(target=drain, args=(enc, "encode-stderr"), daemon=True)
        tr.start()
        tw.start()
        td1.start()
        td2.start()

        # 提前把两个后端按最终尺寸建好并预热。
        # 必须赶在 VSR 跑起来之前做：两者在 5K 下的激活峰值分别约 10 GiB / 6 GiB，
        # 虽然生命周期不重叠，但先建好可以避开缓存分配器的碎片问题；
        # 预热则把 cudnn 算法选型的一次性开销挪出正式计时。
        _phase("探元数据/建线程")
        self.vsr.warmup(H0, W0)
        _phase("VSR 预热")
        if factor == 2:
            import torch
            ph = (H1 + self.rife.PAD - 1) // self.rife.PAD * self.rife.PAD
            pw = (W1 + self.rife.PAD - 1) // self.rife.PAD * self.rife.PAD
            torch.cuda.empty_cache()
            self.rife.build(ph, pw)
            _phase("RIFE 建后端")
            self._log(f"RIFE 后端就绪 {pw}x{ph}（batch={self.rife.batch} "
                      f"fp16={self.rife.fp16} graph={self.rife.graph is not None}）")

        # ---- 主循环：微批过 VSR -> RIFE ----
        carry = None                # 上一微批最后一帧（补齐插帧的帧对与连续性）
        produced = 0
        first_emit_of_batch = True
        try:
            pg.stage = "vsr"
            _n_batch = 1
            eof = False
            import torch                      # 主循环里要用 empty_cache

            def _mark(key, dt):
                pg.marks[key] = round(pg.marks.get(key, 0.0) + dt, 1)

            while True:
                if should_stop is not None and should_stop():
                    raise RuntimeError("cancelled")
                # 等解码（上游供不上）
                t_d = time.time()
                batch = []
                while len(batch) < self.micro_batch:
                    # ⚠️ 必须记住"已经收到过 EOF 哨兵"。
                    #
                    # 这里踩过一个很难查的死锁：哨兵会被**最后一批不完整的帧
                    # 连在一起吃掉**。例如 16 帧 / micro_batch=12：
                    #   第1批取满 12 帧；第2批取到 4 帧后又取到哨兵 → 跳出内层。
                    #   处理完第2批，循环回到这里再 qin.get() —— 再也不会有第二个
                    #   哨兵，于是永久阻塞（栈上：主线程卡 qin.get、writer 卡
                    #   qout.get、reader 已退出，看起来像"都闲着"）。
                    # 帧数正好整除时哨兵会留到下一轮被单独取到，所以只在
                    # **不整除**的配置上暴露（128/8 正常，16/12 卡死）。
                    if eof:
                        break
                    item = qin.get()
                    if item is None:
                        eof = True
                        break
                    batch.append(item)
                _mark("wait_decode_s", time.time() - t_d)
                if not batch:
                    break
                src_arr = np.stack(batch)
                t0 = time.time()
                up = self.vsr.upscale_chunk(src_arr)
                _vsr_dt = time.time() - t0
                self._log(f"    微批#{_n_batch} VSR {len(src_arr)}帧 -> {_vsr_dt:.2f}s "
                          f"({_vsr_dt/len(src_arr)*1000:.0f} ms/帧)")
                _n_batch += 1
                _mark("vsr_s", _vsr_dt)
                # 按需释放显存缓存池。
                #
                # 背景：BasicVSR++ 在 2880x5120 输出下 memory_reserved 高达 15.26 GiB
                # （16 GiB 卡的 94%），再建 RIFE 就 18.11 GiB > 物理显存 → 驱动换页
                # → VSR 从 928 ms/帧 崩到 4476 ms/帧。当时必须每轮释放。
                #
                # 但在目标尺寸（1440x2560 输出）下 VSR 只占 3 GiB，压力不存在了，
                # 白释放要付 ~4% 的代价。所以改成**按需**：只在预留量确实偏高时才释放。
                if torch.cuda.memory_reserved() > self.cache_flush_gb * 2**30:
                    torch.cuda.empty_cache()

                if factor == 2:
                    pg.stage = "rife"
                    seq = up if carry is None else np.concatenate([carry[None], up], 0)
                    t1 = time.time()
                    mids = self._interp_pairs(seq)
                    _mark("rife_s", time.time() - t1)
                    frames = [seq[0]]
                    for i in range(len(seq) - 1):
                        frames.append(mids[i])
                        frames.append(seq[i + 1])
                    if carry is not None:
                        frames = frames[1:]      # 首帧是上一批已发出的，跳过
                    # 末尾补一帧，保证「帧数 = 2N」严格成立
                    if len(frames) < 2 * len(batch) and len(up) == len(batch):
                        frames.append(seq[-1])
                    carry = up[-1].copy()
                    if torch.cuda.memory_reserved() > self.cache_flush_gb * 2**30:
                        torch.cuda.empty_cache()
                    t2 = time.time()
                    for fr in frames:
                        _put(qout, fr)
                    # 这一段是等编码器消化（背压），不是计算
                    _mark("wait_encode_s", time.time() - t2)
                    produced += len(frames)
                else:
                    t2 = time.time()
                    for fr in up:
                        _put(qout, fr)
                    _mark("wait_encode_s", time.time() - t2)
                    produced += len(up)
                # 进度要放在**真正交出去之后**再记。放在前面的话，
                # 一旦下游卡住，页面就会显示 100% 但任务永远不结束。
                pg.done = min(pg.done + len(batch), n_src)
                del batch, src_arr, up
            _phase("主循环（含 RIFE）")
        finally:
            # 非阻塞地投结束哨兵：消费者可能已经退出，此时 put 会永久阻塞
            try:
                qout.put_nowait(None)
            except queue.Full:
                while not qout.empty():
                    try:
                        qout.get_nowait()
                    except queue.Empty:
                        break
                try:
                    qout.put_nowait(None)
                except queue.Full:
                    pass
            tw.join(timeout=600)
            if tw.is_alive():
                errors.append("writer 线程未能在 600s 内退出")
            tr.join(timeout=10)
            # ⚠️ 关掉 stdin 后编码器还要 flush 尾部、写 moov box，需要一点时间。
            # 这里直接 poll() 判活然后 kill，会把还没写完的 mp4 截成 0 字节。
            for p in (enc, dec):
                try:
                    p.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    self._log(f"进程 {p.pid} 超时未退出，强制结束")
                    p.kill()
                    p.wait()
            td1.join(timeout=5)
            td2.join(timeout=5)
            for p in (dec, enc):
                try:
                    if p.stderr:
                        p.stderr.close()
                except Exception:                              # noqa: BLE001
                    pass

        _phase("收尾/等编码器退出")
        # ---- 结果校验：宁可报失败，也不要交付一个打不开的文件 ----
        # 之前那次卡死，页面显示 100% 却下载不了，产物其实是缺 moov atom 的残片。
        if enc.returncode not in (0, None):
            errors.append(f"ffmpeg 编码器退出码 {enc.returncode}")
        if written[0] == 0:
            errors.append("没有任何帧写入编码器")
        if not self._probe_ok(vid_only):
            errors.append("产物无法解析（缺少 moov atom，说明编码没写完）")
        if errors:
            tail = "; ".join(errors[-4:])
            for p in (vid_only, out):
                if os.path.exists(p):
                    try:
                        os.remove(p)        # 残片留着只会让人误以为成功了
                    except OSError:
                        pass
            raise RuntimeError(f"流水线失败: {tail}")

        # ---- 合并源视频的音轨 ----
        if self._has_audio(ff, src):
            mux = [ff, "-y", "-v", "error", "-i", vid_only, "-i", src,
                   "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                   "-shortest", "-movflags", "+faststart", out]
            r = subprocess.run(mux, capture_output=True, text=True, errors="replace")
            if r.returncode == 0 and self._probe_ok(out):
                os.remove(vid_only)
            else:
                # 合音轨失败不能把整个任务搞挂 —— 退化成无声版并如实记录
                self._log(f"合音轨失败（{r.stderr.strip()[:120]}），输出无声版")
                os.replace(vid_only, out)
        else:
            os.replace(vid_only, out)

        _phase("产物校验")
        if pg:
            pg.stage = "done"
            pg.done = pg.total
            pg.marks["total_s"] = round(time.time() - pg.t0, 1)
        rep = {
            "output": out,
            "size_mb": round(os.path.getsize(out) / 1048576, 1) if os.path.exists(out) else 0,
            "frames_in": n_src, "frames_out": written[0],
            "fps_src": fps_src, "fps_out": fps_out,
            "resolution": [W1, H1],
            "duration_s": round(out_dur, 3), "source_duration_s": round(dur_src, 3),
            "interp": factor == 2, "realtime": realtime,
            "color": {k: meta.get(k) for k in
                      ("color_primaries", "color_trc", "colorspace", "color_range")},
            "elapsed_s": round(time.time() - pg.t0, 1),
            "stages_s": pg.marks,
        }
        return rep

    # ------------------------------------------------------------------ #
    def _interp_pairs(self, seq: np.ndarray) -> np.ndarray:
        """seq: (N,4H,4W,3) uint8 → 中间帧 (N-1,4H,4W,3) uint8。"""
        import torch
        N = len(seq)
        if N < 2:
            return np.zeros((0,) + seq.shape[1:], np.uint8)
        r = self.rife
        pad = r.PAD
        h, w = seq.shape[1], seq.shape[2]
        ph = (h + pad - 1) // pad * pad
        pw = (w + pad - 1) // pad * pad
        r.build(ph, pw)
        dev = r.device
        dt = r.dtype
        # 先整段搬到显存（uint8 中转，省内存），再按 batch 切
        a_np = np.ascontiguousarray(seq[:-1])
        b_np = np.ascontiguousarray(seq[1:])
        outs = []
        bs = r.batch
        for i in range(0, N - 1, bs):
            ja, jb = a_np[i:i + bs], b_np[i:i + bs]
            ta = torch.from_numpy(ja).to(dev, non_blocking=True).permute(0, 3, 1, 2)
            tb = torch.from_numpy(jb).to(dev, non_blocking=True).permute(0, 3, 1, 2)
            ta = ta.float().div_(255.0)
            tb = tb.float().div_(255.0)
            if ph != h or pw != w:
                ta = torch.nn.functional.pad(ta, (0, pw - w, 0, ph - h), mode="replicate")
                tb = torch.nn.functional.pad(tb, (0, pw - w, 0, ph - h), mode="replicate")
            ta, tb = ta.to(dt), tb.to(dt)
            if ta.shape[0] != r.batch:
                pad_n = r.batch - ta.shape[0]
                ta = torch.cat([ta, ta[-1:].expand(pad_n, -1, -1, -1)], 0)
                tb = torch.cat([tb, tb[-1:].expand(pad_n, -1, -1, -1)], 0)
            y = r.infer(ta, tb)
            # 模型输出是 inference tensor，出了 inference_mode 只能做**非原地**运算
            y = y[:ja.shape[0], :, :h, :w].float().clamp(0, 1).mul(255.0).round()
            outs.append(y.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())
            del ta, tb, y
        return np.concatenate(outs, 0) if len(outs) > 1 else outs[0]
