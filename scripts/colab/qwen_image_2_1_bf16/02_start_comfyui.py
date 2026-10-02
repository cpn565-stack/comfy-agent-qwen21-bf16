"""Start this kit's BF16 ComfyUI server and cloudflared from a Colab cell."""

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

WORKSPACE = Path(os.environ.get("QWEN21_BF16_WORKSPACE", "/content/ComfyUI-Qwen21-BF16"))
PORT = int(os.environ.get("QWEN21_BF16_PORT", "8188"))
URL_FILE = Path("/content/qwen21_bf16_url.txt")
LOG_FILE = Path("/content/qwen21_bf16_comfy.log")
TUNNEL_LOG = Path("/content/qwen21_bf16_tunnel.log")
BF16_FLAGS = ["--bf16-unet", "--bf16-text-enc", "--bf16-vae"]


def start():
    manifest = json.loads((WORKSPACE / "qwen21-bf16-profile.json").read_text())
    if manifest.get("profile") != "qwen_image_2_1_bf16":
        raise RuntimeError("Run this kit's BF16 setup before starting ComfyUI.")
    probe = """
import re, torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
p = torch.cuda.get_device_properties(0)
if not re.search(r'\\bA100\\b', p.name, re.I):
    raise RuntimeError('A100 required: ' + p.name)
if p.total_memory < 38 * 1024**3:
    raise RuntimeError('A100 40GB/80GB required')
if not torch.cuda.is_bf16_supported():
    raise RuntimeError('BF16 unavailable')
"""
    nvidia_dir = "/usr/lib64-nvidia"
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if Path(nvidia_dir).is_dir() and nvidia_dir not in paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([nvidia_dir] + [p for p in paths if p])
    subprocess.run([sys.executable, "-c", probe], check=True)
    for name, size in manifest["model_files"].items():
        path = WORKSPACE / "models" / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Missing/incomplete BF16 model: {name}. Re-run setup.")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"Port {PORT} is occupied. Stop the previous server or set QWEN21_BF16_PORT.")
    URL_FILE.unlink(missing_ok=True)

    # Normal ComfyUI memory management may offload to RAM; it does not quantize.
    with LOG_FILE.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "main.py", *BF16_FLAGS, "--cache-none",
             "--dont-print-server", "--listen", "127.0.0.1", "--port", str(PORT)],
            cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
        )

    def tunnel():
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                print(f"ComfyUI exited. Check {LOG_FILE}.")
                return
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", PORT)) == 0:
                    break
            time.sleep(0.5)
        else:
            print(f"ComfyUI did not open port {PORT}; check {LOG_FILE}.")
            return
        process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        with TUNNEL_LOG.open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
                if match and not URL_FILE.exists():
                    URL_FILE.write_text(match.group(0))
                    print(f"Ready URL: {match.group(0)}")
                    print(f"On the local Mac: comfy-agent connect {match.group(0)}")

    threading.Thread(target=tunnel, daemon=True).start()
    print(f"BF16 ComfyUI started (PID {server.pid}); URL file: {URL_FILE}; log: {LOG_FILE}")
    print("This launcher uses a Colab/Jupyter kernel thread for the tunnel. Keep the kernel running.")


if __name__ == "__main__":
    start()
