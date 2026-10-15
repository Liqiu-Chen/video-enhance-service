"""MP4/MOV 色彩元数据解析。

环境里没有 ffprobe，且 ffmpeg 的 `-i` 输出只能给出矩阵（matrix），
拿不到 transfer / primaries 的权威值，因此直接解析 MP4 的 colr box。

参考：ISO/IEC 14496-12（容器）、ISO/IEC 23001-8（colr 取值）。
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
from typing import Dict, List, Optional, Tuple

# ---- colr box 取值 → ffmpeg 名称 -------------------------------------------
PRIMARIES = {
    1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
    8: "film", 9: "bt2020", 10: "smpte428", 11: "smpte431", 12: "smpte432",
    22: "ebu3213",
}
TRANSFER = {
    1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
    8: "linear", 11: "iec61966-2-4", 13: "iec61966-2-1", 14: "bt2020-10",
    15: "bt2020-12", 16: "smpte2084", 18: "arib-std-b67",
}
MATRIX = {
    0: "gbr", 1: "bt709", 5: "bt470bg", 6: "smpte170m", 8: "ycgco",
    9: "bt2020nc", 10: "bt2020c",
}

# 需要递归进入的容器 box
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}
# 需要跳过的 4 字节 prefixed 容器
_STD_ENTRIES = {b"avc1", b"hvc1", b"hev1", b"mp4v", b"av01", b"vp09"}


def _iter_boxes(f, start: int, end: int):
    """在 [start, end) 区间内迭代顶层 box，产出 (type, data_start, box_end)。"""
    f.seek(start)
    while f.tell() + 8 <= end:
        pos = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size = int.from_bytes(hdr[:4], "big")
        typ = hdr[4:8]
        if size == 1:                       # 64 位 largesize
            size = int.from_bytes(f.read(8), "big")
        elif size == 0:                     # 延伸到文件末尾
            size = end - pos
        if size < 8 or pos + size > end + 1:
            return
        yield typ, pos + (16 if size > 8 and hdr[4:8] and int.from_bytes(hdr[:4], "big") == 1 else 8), pos + size
        f.seek(pos + size)


def _find_colr_in_stsd(f, stsd_data: int, stsd_end: int) -> List[dict]:
    """stsd 里每个 sample entry 的 78 字节之后是子 box，colr 就在这里。"""
    out: List[dict] = []
    f.seek(stsd_data + 8)                   # 跳过 fullbox version/flags + entry_count
    while f.tell() + 8 <= stsd_end:
        pos = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        esize = int.from_bytes(hdr[:4], "big")
        etype = hdr[4:8]
        if esize < 8:
            break
        if etype in _STD_ENTRIES:
            child = pos + 8 + 78            # VisualSampleEntry 固定头长度
            for ctyp, cdata, cend in _iter_boxes(f, child, pos + esize):
                if ctyp == b"colr":
                    f.seek(cdata)
                    payload = f.read(cend - cdata)
                    out.append(_parse_colr(payload))
                elif ctyp == b"pasp":
                    f.seek(cdata)
                    h_sp, v_sp = struct.unpack(">II", f.read(8))
                    out.append({"box": "pasp", "sar": f"{h_sp}:{v_sp}"})
        f.seek(pos + esize)
    return out


def _parse_colr(payload: bytes) -> dict:
    ctype = payload[:4].decode("latin-1", "replace")
    d = {"box": "colr", "colour_type": ctype}
    if ctype in ("nclx", "nclc") and len(payload) >= 10:
        p, t, m = struct.unpack(">HHH", payload[4:10])
        d.update(primaries_id=p, transfer_id=t, matrix_id=m)
        if ctype == "nclx" and len(payload) >= 11:
            d["full_range"] = bool(payload[10] & 0x80)
        d["color_primaries"] = PRIMARIES.get(p)
        d["color_trc"] = TRANSFER.get(t)
        d["colorspace"] = MATRIX.get(m)
    return d


def parse_mp4_colr(path: str) -> dict:
    """从 MP4/MOV 里挖出 colr / pasp box 的原始信息。"""
    size = os.path.getsize(path)
    found: dict = {}
    with open(path, "rb") as f:
        stack = [(0, size)]
        # 只走 moov 分支：先定位 moov
        moov = None
        for typ, dstart, dend in _iter_boxes(f, 0, min(size, 1 << 24)):
            if typ == b"moov":
                moov = (dstart, dend)
                break
        if not moov:
            return {"error": "moov not found (可能不是标准 MP4/MOV)"}
        # moov -> trak -> mdia -> minf -> stbl -> stsd
        for t1, d1, e1 in _iter_boxes(f, *moov):
            if t1 != b"trak":
                continue
            hdlr = None
            for t2, d2, e2 in _iter_boxes(f, d1, e1):
                if t2 == b"mdia":
                    for t3, d3, e3 in _iter_boxes(f, d2, e2):
                        if t3 == b"hdlr" and hdlr is None:
                            f.seek(d3 + 8)
                            hdlr = f.read(4)
                        if t3 == b"minf":
                            for t4, d4, e4 in _iter_boxes(f, d3, e3):
                                if t4 == b"stbl":
                                    for t5, d5, e5 in _iter_boxes(f, d4, e4):
                                        if t5 == b"stsd":
                                            found.update(
                                                {k: v for k, v in
                                                 ((x.get("box"), x) for x in
                                                  _find_colr_in_stsd(f, d5, e5))
                                                 if k})
                                            if "colr" in found:
                                                found["track_handler"] = (
                                                    hdlr.decode("latin-1") if hdlr else None)
                                                return found
    return found or {"note": "未找到 colr box（源文件未标注色彩信息）"}


# ---- ffmpeg -i 解析 ---------------------------------------------------------
_STREAM_RE = re.compile(
    r"Stream #\d+:\d+(?:\[[^\]]*\])?(?:\([^)]*\))?: Video: (?P<codec>\w+).*?"
    r"(?P<pixfmt>[a-z0-9]+)\((?P<attrs>[^)]*)\)\s*,\s*(?P<w>\d+)x(?P<h>\d+)"
    r".*?,\s*(?P<fps>[\d.]+)\s*fps", re.S)
_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")


def probe_with_ffmpeg(ffmpeg: str, path: str) -> dict:
    """用 ffmpeg -i 拿容器/流信息（无 ffprobe 时的替代方案）。"""
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", path],
                       capture_output=True, text=True, errors="replace")
    txt = p.stderr
    info: dict = {}
    m = _DUR_RE.search(txt)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration"] = hh * 3600 + mm * 60 + ss
    m = _STREAM_RE.search(txt)
    if not m:
        info["raw_stderr_tail"] = txt[-800:]
        return info
    info.update(codec=m.group("codec"), pix_fmt=m.group("pixfmt"),
                width=int(m.group("w")), height=int(m.group("h")),
                fps=float(m.group("fps")))
    attrs = [a.strip() for a in m.group("attrs").split(",")]
    # ffmpeg 打印格式为 pix_fmt(range, matrix, progressive) 或
    # pix_fmt(range, matrix/primaries/transfer, progressive)
    prog = attrs[-1] if attrs and attrs[-1] == "progressive" else None
    attrs = [a for a in attrs if a != "progressive"]
    if attrs and attrs[0] in ("tv", "pc"):
        info["color_range"] = attrs.pop(0)
    if attrs:
        parts = attrs[0].split("/")
        if len(parts) == 1:
            info["colorspace"] = parts[0]
        elif len(parts) == 3:
            info["colorspace"], info["color_primaries"], info["color_trc"] = parts
    if prog:
        info["scan"] = prog
    info["has_audio"] = "Audio:" in txt
    return info


def color_report(ffmpeg: str, path: str) -> dict:
    """合并两路信息，给出最终用于解码/编码的色彩参数。"""
    colr = parse_mp4_colr(path)
    ff = probe_with_ffmpeg(ffmpeg, path)
    merged = dict(ff)
    src = "ffmpeg"
    colr_entry = colr.get("colr") if isinstance(colr, dict) else None
    if isinstance(colr_entry, dict):
        src = "colr-box"
        for k in ("color_primaries", "color_trc", "colorspace"):
            if colr_entry.get(k):
                merged[k] = colr_entry[k]
        if "full_range" in colr_entry:
            merged["color_range"] = "pc" if colr_entry["full_range"] else "tv"
    if isinstance(colr, dict) and colr.get("sar"):
        merged["sar"] = colr["sar"]
    # 记录"文件里真正声明了哪些字段"，用于区分「源未标注」与「真的不一致」
    declared = [k for k in ("color_primaries", "color_trc", "colorspace", "color_range")
                if merged.get(k)]
    merged["_declared"] = declared
    # SDR 8bit 的兜底推断：矩阵/传输/原色都缺省时按 bt709 处理（与播放器一致）
    if str(merged.get("pix_fmt", "")).startswith(("yuv", "nv")):
        depth10 = any(t in str(merged.get("pix_fmt", "")) for t in ("10le", "10be", "12le"))
        default_cs = "bt2020nc" if depth10 else "bt709"
        merged.setdefault("colorspace", default_cs)
        if not depth10:
            merged.setdefault("color_primaries", "bt709")
            merged.setdefault("color_trc", "bt709")
    merged["_source"] = src
    merged["_colr_raw"] = colr
    return merged


def setparams_filter(meta: dict) -> Optional[str]:
    """构造 setparams 滤镜串。

    坑：输入是纯 RGB（PNG 序列）时，ffmpeg 的滤镜链会把"输入侧未知"的
    primaries/transfer 一路带到输出，即使命令行给了 -color_primaries 也**不会**
    写进输出文件（matrix 却能写进去）。必须在滤镜链里显式 setparams，
    编码器才能拿到这些属性，MP4 封装器才会写出 colr box。
    """
    parts = []
    for key in ("color_primaries", "color_trc", "colorspace"):
        v = meta.get(key)
        if v and v != "unknown":
            parts.append(f"{key}={v}")
    r = meta.get("color_range")
    if r and r != "unknown":
        parts.append("range=" + ("limited" if r == "tv" else "full"))
    return "setparams=" + ":".join(parts) if parts else None


def ffmpeg_args(meta: dict) -> List[str]:
    """把色彩参数翻译成 ffmpeg 命令行开关（用于编码时标注输出）。"""
    args: List[str] = []
    mapping = [("color_primaries", "-color_primaries"),
               ("color_trc", "-color_trc"),
               ("colorspace", "-colorspace"),
               ("color_range", "-color_range")]
    valid = {"color_primaries": set(PRIMARIES.values()), "color_trc": set(TRANSFER.values()),
             "colorspace": set(MATRIX.values()), "color_range": {"tv", "pc", "limited", "full"}}
    for key, flag in mapping:
        v = meta.get(key)
        if v and (key not in valid or v in valid[key]):
            args += [flag, "limited" if (key == "color_range" and v == "tv")
                     else ("full" if (key == "color_range" and v == "pc") else v)]
    return args


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    import imageio_ffmpeg
    print(json.dumps(color_report(imageio_ffmpeg.get_ffmpeg_exe(), sys.argv[1]),
                     ensure_ascii=False, indent=2))


def true_fps(ffmpeg: str, path: str, sample: int = 400) -> float | None:
    """用帧间隔的众数反推真实帧率，避免被容器的 avg_frame_rate 误导。

    为什么要这样：容器 Duration 取的是**最长轨道**的长度。若音轨比视频轨长
    （实测某竖屏片：视频 791 帧@60fps = 13.19s，音轨 14.74s），
    ffmpeg 报告的 avg_frame_rate 会被摊薄成 53.92，
    按它编码会把视频拉长 11%。用相邻帧 pts 间隔的众数才准。
    """
    try:
        p = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", path, "-vf", "showinfo",
             "-fps_mode", "passthrough", "-frames:v", str(sample), "-f", "null", "-"],
            capture_output=True, text=True, errors="replace")
        pts = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]
    except Exception:
        return None
    if len(pts) < 4:
        return None
    deltas = [round(pts[i + 1] - pts[i], 5) for i in range(len(pts) - 1)]
    deltas = [d for d in deltas if d > 1e-6]
    if not deltas:
        return None
    # 众数
    best, cnt = 0.0, {}
    for d in deltas:
        cnt[d] = cnt.get(d, 0) + 1
        if cnt[d] > cnt.get(best, 0):
            best = d
    fps = 1.0 / best
    # 归到最接近的标准帧率（取"最接近"而不是"第一个满足容差"，否则 59.99 会错配成 59.94）
    stds = (23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120)
    near = min(stds, key=lambda s: abs(fps - s))
    return float(near) if abs(fps - near) < 0.05 else round(fps, 3)
