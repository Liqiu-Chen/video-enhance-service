"""启动推理服务。

    python serve.py                    # 监听 0.0.0.0:8000（局域网可访问）
    python serve.py --port 9000
    python serve.py --host 127.0.0.1   # 只允许本机
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0=局域网可访问")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None,
                    help="访问令牌；不传则用环境变量 INFER_TOKEN，再没有就随机生成")
    args = ap.parse_args()
    if args.token:
        os.environ["INFER_TOKEN"] = args.token

    import app as appmod          # noqa: PLC0415  —— 取到最终生效的令牌
    token = appmod.TOKEN
    ip = lan_ip()
    print("=" * 66)
    print("  VSR ×4 + RIFE ×2 推理服务")
    print("=" * 66)
    print(f"  本机网页：     http://127.0.0.1:{args.port}/?token={token}")
    print(f"  局域网访问：   http://{ip}:{args.port}/?token={token}")
    print()
    print(f"  访问令牌：     {token}")
    print(f"  接口文档：     http://{ip}:{args.port}/docs")
    print()
    print("  ⚠️ 服务无用户体系，令牌就是唯一凭据，不要贴到公开场合。")
    print("  ⚠️ 想把地址发给局域网外的用户，需要另外做内网穿透（见 README 第七节）。")
    print("=" * 66, flush=True)

    import uvicorn
    uvicorn.run(appmod.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
