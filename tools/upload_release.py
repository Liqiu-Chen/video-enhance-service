"""把 VSR 的 TF32 引擎作为 GitHub Release 资产上传。

为什么必须用 Python 而不是 curl/git：
  本机 Windows 自带的 schannel TLS 后端处理不了 GitHub 触发的 TLS 重协商
  （curl -v 会看到 "remote party requests renegotiation" 死循环）。
  Python 用的是 OpenSSL，完全正常。

为什么走 Release 而不是 git：
  GitHub 单文件上限 100 MB，而引擎是 135 MB / 156 MB。

用法：
    python tools/upload_release.py                 # 上传默认那个 t=8 引擎
    python tools/upload_release.py --all           # 连 t=2 一起传
    python tools/upload_release.py --tag v1.0.1    # 换 tag
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 引擎可能在这几个位置（开发时在主项目 trt5/，独立部署时放 engines/）
_SEARCH = [
    os.path.join(ROOT, "engines"),            # 本仓库 engines/
    os.path.join(ROOT, "trt5"),               # 本仓库 trt5/
    os.path.join(os.path.dirname(ROOT), "trt5"),   # 上一级（开发布局）
    ROOT,
]


def find_engine(rel: str):
    """在候选目录里找引擎文件，返回绝对路径或 None。"""
    base = os.path.basename(rel)
    for d in _SEARCH:
        for cand in (os.path.join(d, base), os.path.join(d, rel)):
            if os.path.isfile(cand):
                return cand
    return None

TOKEN = os.environ.get("GITHUB_TOKEN", "")
OWNER, REPO = "Liqiu-Chen", "video-enhance-service"
PROXY = os.environ.get("FI_PROXY", "http://127.0.0.1:7897")

# (文件名, 一句话说明)
ASSETS = {
    "t8": ("trt5/vsr_tf32_t8_836x480_x4.trt",
           "VSR 主引擎 · TensorRT TF32 · t=8 · 135 MB"),
    "t2": ("trt5/vsr_tf32_t2_836x480_x4.trt",
           "VSR 引擎（t=2，调试/短片段用）· 156 MB"),
}

NOTES = """## VSR 超分引擎（TensorRT TF32）

面向 BasicVSR++ ×4 的 TensorRT 引擎。**引擎是固定形状的**——`t` 是编译时
写进图里的帧数，输入分辨率也是固定的。

### 实测数据

对 PyTorch fp32 真值、真实视频帧 480×836 → ×4、两边都不分块：

| 实现 | ms/帧 | 相对 PyTorch fp32 | PSNR | 最大差 | >2/255 |
|---|---|---|---|---|---|
| PyTorch fp32（基准） | 441.9 | 1.00× | — | 0 | 0% |
| PyTorch fp16（autocast） | — | — | 99.00 dB | 0.0 | 0% |
| **TensorRT TF32（本资产）** | **260.9** | **1.69×** | **78.54 dB** | **1.0** | **0.000%** |
| TensorRT fp16（未发布） | 93.3 | 3.97× | 58.93 dB | 21.0 | 0.166% |

**只保留 TF32 版本**：它精度最高（比 fp32 引擎的 72.22 dB 还高），
且比 PyTorch fp32 快 1.69×。fp16 那版虽然快 4×，但最大差 21/255 不达标。

### 怎么用

```bash
# 1. 下载后放到仓库根的 trt5/ 目录
mkdir -p trt5 && mv vsr_tf32_t8_836x480_x4.trt trt5/

# 2. 服务里选它
#    sr_backend="vsr-trt"（VsrTrtEngine 默认 mode="tf32"）
python service/serve.py --host 0.0.0.0 --port 8000
```

### ⚠️ 两个使用注意

1. **`t` 必须和 `vsr_chunk` 匹配。** 本引擎是 `t=8`，配合默认
   `vsr_chunk=8` 使用。若序列长度不等于 `t`，适配器会在尾部重复最后一帧补齐——
   **这不是无损的**（实测 58.22 dB 偏差），因为 BasicVSR++ 有反向传播路径。

2. **换分辨率或换 `t` 都要重新编译**（约 4~5 分钟）。
   构建方式见 `engines/README.md`，其中最关键的一步：
   **导出 ONNX 时必须用真实视频帧做 dummy，绝不能用 `torch.zeros`** ——
   否则会触发 `is_mirror_extended` 分支，导出一个缺了正向光流的残缺图
   （精度只剩 31 dB，且这个问题极难从现象反推）。
"""


def opener():
    h = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    return urllib.request.build_opener(
        h, urllib.request.HTTPSHandler(context=ssl.create_default_context()))


def call(path: str, method="GET", data=None, host="api.github.com",
         raw=None, content_type="application/json"):
    url = f"https://{host}{path}"
    body = raw if raw is not None else (
        json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, method=method, data=body, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "video-enhance-service-uploader",
        "Content-Type": content_type,
    })
    try:
        with opener().open(req, timeout=1800) as r:
            txt = r.read()
            try:
                return r.status, json.loads(txt or b"{}")
            except Exception:
                return r.status, {}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1.0.0")
    ap.add_argument("--all", action="store_true", help="连 t=2 一起传")
    ap.add_argument("--only", choices=list(ASSETS), default="t8")
    args = ap.parse_args()
    if not TOKEN:
        print("❌ 需要 GITHUB_TOKEN 环境变量")
        return 1

    keys = list(ASSETS) if args.all else [args.only]
    files = []
    for k in keys:
        rel, desc = ASSETS[k]
        found = find_engine(rel)
        if found:
            files.append((k, found, desc))
        else:
            print(f"  ⚠️ 找不到 {os.path.basename(rel)}，已跳过")
    if not files:
        print("❌ 一个引擎都没找到。搜索过：")
        for d in _SEARCH:
            print(f"     {d}")
        return 1

    print(f"仓库 {OWNER}/{REPO}    tag {args.tag}")
    for k, path, desc in files:
        print(f"  待传 {os.path.basename(path):<34} {os.path.getsize(path)/2**20:6.1f} MB")

    # 1) 建 release（tag 不存在会自动创建）
    st, rel = call(f"/repos/{OWNER}/{REPO}/releases", "POST", {
        "tag_name": args.tag,
        "name": f"{args.tag} — VSR TensorRT TF32 引擎",
        "body": NOTES,
        "draft": False, "prerelease": False,
    })
    if st == 422:            # 已存在 → 复用
        st2, rel = call(f"/repos/{OWNER}/{REPO}/releases/tags/{args.tag}")
        if st2 != 200:
            print(f"❌ 取已有 release 失败: {rel.get('message')}")
            return 1
        print(f"  release {args.tag} 已存在，复用 (id={rel['id']})")
    elif st not in (200, 201):
        print(f"❌ 建 release 失败 HTTP {st}: {rel.get('message')}")
        return 1
    else:
        print(f"  ✅ release 已建 (id={rel['id']})")

    rid = rel["id"]
    st, existing = call(f"/repos/{OWNER}/{REPO}/releases/{rid}/assets")
    have = {a["name"]: a["id"] for a in (existing if isinstance(existing, list) else [])}

    # 2) 上传资产
    ok = True
    for k, p, desc in files:
        name = os.path.basename(p)
        if name in have:
            print(f"  ⏭  {name} 已存在，跳过")
            continue
        size = os.path.getsize(p)
        print(f"  ⬆ 上传 {name} ({size/2**20:.1f} MB)...", flush=True)
        t0 = time.time()
        # uploads.github.com 是独立域名，资产走它
        st, d = call(f"/repos/{OWNER}/{REPO}/releases/{rid}/assets?name={name}",
                     "POST", raw=open(p, "rb").read(),
                     host="uploads.github.com",
                     content_type="application/octet-stream")
        if st in (200, 201):
            print(f"     ✅ 完成 {time.time()-t0:.0f}s  "
                  f"下载: {d.get('browser_download_url')}")
        else:
            print(f"     ❌ HTTP {st}: {d.get('message')}")
            ok = False

    print()
    print(f"  Release 页面：https://github.com/{OWNER}/{REPO}/releases/tag/{args.tag}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
