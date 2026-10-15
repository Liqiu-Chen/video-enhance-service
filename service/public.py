"""把本地推理服务通过免注册 SSH 隧道发布到公网，自动重连、并把**当前地址**写到文件。

    python public.py

为什么用 SSH 隧道而不是 cloudflared / ngrok：
本机实测 **github.com 和 cloudflare.com 的 TCP 连接被阻断**，cloudflared 既下载不到
也连不上边缘节点；ngrok / localtunnel / pinggy 全不通；serveo / bore 现在都要公钥。
只剩 localhost.run 能免注册用起来。

⚠️ **免费隧道的地址是临时的**：SSH 一断线重连就换域名（实测几分钟就会断一次）。
所以这个脚本做三件事把影响降到最低：

1. 把当前地址写进 `current_url.txt`；服务端有 `/api/tunnel` 接口读它，
   **网页页头会显示当前公网地址** —— 局域网地址永不改变，
   所以从局域网打开页面就总能拿到最新的公网地址。
2. 起一个**存活探测线程**：定期探公网地址，发现不是 200/401 就立刻主动重连，
   而不是傻等 SSH 自己发现。
3. 加强 keepalive（TCPKeepAlive + ServerAliveInterval=10），尽量少断。
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
URL_FILE = os.path.join(HERE, "current_url.txt")
URL_RE = re.compile(r"https://[a-z0-9]+\.lhr\.life")

_state: dict = {"url": None, "proc": None, "relaunch": False}


def _write_url(url: str | None):
    _state["url"] = url
    try:
        tmp = URL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(url or "")
        os.replace(tmp, URL_FILE)          # 原子替换，避免读到半截
    except OSError:
        pass


def _watchdog(fail_needed: int = 3, interval: int = 20):
    """定期探活。**必须连续失败 fail_needed 次才重连。**

    ⚠️ 这里踩过一个坑：最初写成"一次探测失败就立刻重连"，结果
    隧道只要有一次瞬时抖动就被自己掐断，而**重连必然换域名** ——
    等于把还能用的隧道反复杀掉、地址反复变，比不探活还糟。
    短暂 502 对用户来说只是刷新一下，换地址才是真正的麻烦。
    """
    fails = 0
    while True:
        time.sleep(interval)
        url = _state["url"]
        if not url:
            fails = 0
            continue
        code = ""
        try:
            p = subprocess.run(
                ["curl", "-s", "-o", os.devnull, "-m", "25", "-w", "%{http_code}",
                 f"{url}/api/health"],
                capture_output=True, text=True, timeout=40)
            code = (p.stdout or "").strip()
        except Exception:                      # noqa: BLE001
            code = ""
        # 401 也算活着（只是没带令牌）；000 = 连不上
        if code in ("200", "401"):
            fails = 0
            continue
        fails += 1
        print(f"[tunnel] 探活失败 {fails}/{fail_needed}（HTTP {code or '无响应'}）",
              flush=True)
        if fails >= fail_needed:
            print(f"[tunnel] 连续 {fails} 次探活失败（约 "
                  f"{fails*interval}s），判定隧道已死，主动重连", flush=True)
            fails = 0
            _state["relaunch"] = True
            proc = _state.get("proc")
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass


def main():
    host = os.environ.get("SERVE_HOST", "127.0.0.1")
    port = int(os.environ.get("SERVE_PORT", "8000"))
    token = os.environ.get("INFER_TOKEN", "（见服务启动日志）")
    key = os.environ.get("TUNNEL_KEY") or os.path.join(HERE, "work", "tunnel_key")
    cmd = ["ssh", "-T",
           "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
           # 加强保活：免费隧道很容易被中间设备掐掉空闲连接
           "-o", "TCPKeepAlive=yes",
           "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
           "-o", "ConnectTimeout=15",
           "-o", "ExitOnForwardFailure=yes"]
    #
    # ★★★ 用户名绝不能用 `nokey` ★★★
    #
    # FAQ 原文：「Free tunnels change domain names after a few hours. To keep your
    # domain name between connections make sure you have an SSH key **and that you
    # are not using the `nokey` username**.」
    #
    # `nokey` 是"跳过密钥校验、走匿名通道"的专用账号 —— 用了它，密钥**永远不会
    # 生效**，域名每次重连都变（实测 1.5 小时换了 7 次）。
    # 换成普通用户名 + 自己的密钥后，域名**跨重连保持不变**（实测两次连接
    # 拿到完全相同的子域）。这就是之前折腾半天的根因。
    if os.path.exists(key):
        cmd += ["-i", key]
        target = os.environ.get("TUNNEL_USER", "localhost.run")
        print(f"[tunnel] 使用密钥 {key}（域名将保持固定）", flush=True)
    else:
        target = "nokey@localhost.run"
        print("[tunnel] 未找到密钥，退回匿名通道（域名每次重连都会变）", flush=True)
    cmd += ["-R", f"80:{host}:{port}", target]

    threading.Thread(target=_watchdog, daemon=True).start()
    print(f"[tunnel] 目标 {host}:{port} → localhost.run", flush=True)

    while True:
        _state["relaunch"] = False
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=0)
        _state["proc"] = p
        seen: set = set()
        try:
            for line in iter(p.stdout.readline, b""):
                txt = line.decode("utf-8", "replace").rstrip()
                if txt.strip():
                    print("  " + txt, flush=True)
                m = URL_RE.search(txt)
                if m and m.group(0) not in seen:
                    seen.add(m.group(0))
                    url = m.group(0)
                    _write_url(url)
                    print("\n" + "=" * 64)
                    print("  公网访问地址（当前有效，断线会变）：")
                    print(f"    {url}/?token={token}")
                    print("=" * 64 + "\n", flush=True)
                if _state["relaunch"]:
                    break
        except KeyboardInterrupt:
            p.terminate()
            return
        p.wait()
        _write_url(None)
        print("[tunnel] 连接断开，5 秒后重连（重连后地址会变）", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
