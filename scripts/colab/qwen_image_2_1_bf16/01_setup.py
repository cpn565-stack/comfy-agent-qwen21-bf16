"""Set up non-quantized Qwen Image 2.1 on an A100 Colab runtime.

Paste into a Colab cell or use the official `colab --auth oauth2 exec -f`.
This is ordinary Python, so it also works from the remote SSH shell.
The hardware/resource checks run before installs or model downloads.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

WORKSPACE = Path(os.environ.get("QWEN21_BF16_WORKSPACE", "/content/ComfyUI-Qwen21-BF16"))
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_COMMIT = "1b883beab11c04a2eb82cf6a3ee64294f3303897"
MODEL_REPO = "Comfy-Org/Qwen-Image-2.1"
MODEL_REVISION = "cb504a4090723e43f17ad01cec0359490e2de613"
MODEL_FILES = {
    "diffusion_models/qwen_image_2.1_bf16.safetensors": 14230280616,
    "text_encoders/qwen3vl_8b_bf16.safetensors": 17534334616,
    "vae/qwen_image_2.1_vae_bf16.safetensors": 675509688,
}
GIB = 1024 ** 3
# Conservative limits for this profile, not an upstream minimum-memory claim.
MIN_VRAM_GIB = 38
MIN_RAM_GIB = 48
DISK_HEADROOM_GIB = 10


def run(args):
    subprocess.run([str(arg) for arg in args], check=True)


def torch_snapshot():
    # A fresh child inherits the NVIDIA library path even when run via SSH.
    code = """
import json, torch
info = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "available": torch.cuda.is_available()}
if info["available"]:
    gpu = torch.cuda.get_device_properties(0)
    info.update(name=gpu.name, vram_bytes=gpu.total_memory,
                bf16=torch.cuda.is_bf16_supported())
print(json.dumps(info))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def require_a100(info):
    if not info.get("available"):
        raise RuntimeError("CUDA unavailable. Select an A100 Colab runtime before setup.")
    if not re.search(r"\bA100\b", info.get("name", ""), re.IGNORECASE):
        raise RuntimeError(f"A100 required; assigned GPU is {info.get('name')!r}.")
    if not info.get("bf16"):
        raise RuntimeError("The assigned A100 does not report BF16 support.")
    if info.get("vram_bytes", 0) < MIN_VRAM_GIB * GIB:
        raise RuntimeError("This profile requires an A100 40GB or 80GB, not a small MIG slice.")


def require_resources(ram_bytes, free_disk_bytes, remaining_model_bytes):
    if ram_bytes < MIN_RAM_GIB * GIB:
        raise RuntimeError(
            f"This BF16 profile requires at least {MIN_RAM_GIB} GiB system RAM. "
            "Select a high-memory A100 runtime; GPU VRAM and system RAM are different."
        )
    required = remaining_model_bytes + DISK_HEADROOM_GIB * GIB
    if free_disk_bytes < required:
        raise RuntimeError(f"Need {required / GIB:.1f} GiB free disk for remaining weights and setup.")


def verify_model_file(path, expected_bytes):
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Incomplete model file: {path.name}")
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size <= 16 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header: {path.name}")
        header = json.loads(stream.read(header_size))
    dtypes = {value["dtype"] for key, value in header.items() if key != "__metadata__"}
    if "BF16" not in dtypes or any(dtype in {"I8", "U8"} or dtype.startswith("F8") for dtype in dtypes):
        raise RuntimeError(f"Expected non-quantized BF16 weights: {path.name}; dtypes={sorted(dtypes)}")


def setup():
    if not Path("/content").is_dir() or not sys.platform.startswith("linux"):
        raise RuntimeError("Run this setup in the selected Colab runtime, not on the local Mac.")
    nvidia_dir = "/usr/lib64-nvidia"
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if Path(nvidia_dir).is_dir() and nvidia_dir not in paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([nvidia_dir] + [p for p in paths if p])

    before = torch_snapshot()
    require_a100(before)
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk_root = next(parent for parent in [WORKSPACE, *WORKSPACE.parents] if parent.exists())
    remaining = sum(
        size for name, size in MODEL_FILES.items()
        if not (WORKSPACE / "models" / name).is_file()
        or (WORKSPACE / "models" / name).stat().st_size != size
    )
    require_resources(ram_bytes, shutil.disk_usage(disk_root).free, remaining)
    print(f"Preflight OK: {before['name']}, {before['vram_bytes'] / GIB:.1f} GiB VRAM, "
          f"{ram_bytes / GIB:.1f} GiB RAM; remaining weights {remaining / 1e9:.2f} GB.")

    if not WORKSPACE.exists():
        run(["git", "clone", "--depth", "1", COMFYUI_REPO, WORKSPACE])
    elif not (WORKSPACE / ".git").is_dir():
        raise RuntimeError(f"Workspace exists but is not a ComfyUI checkout: {WORKSPACE}")
    run(["git", "-C", WORKSPACE, "fetch", "--depth", "1", "origin", COMFYUI_COMMIT])
    run(["git", "-C", WORKSPACE, "checkout", "--detach", COMFYUI_COMMIT])
    nodes = (WORKSPACE / "comfy_extras/nodes_qwen.py").read_text()
    if "TextEncodeQwenImage21" not in nodes or "QwenImage21Cache" not in nodes:
        raise RuntimeError("The pinned ComfyUI checkout has no Qwen Image 2.1 nodes.")

    # Keep the working Colab Torch/CUDA stack. Conflicts fail instead of replacing it.
    pins = []
    for package in ["torch", "torchvision", "torchaudio"]:
        try:
            pins.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    with tempfile.TemporaryDirectory(prefix="qwen21-bf16-") as temp_dir:
        constraints = Path(temp_dir) / "constraints.txt"
        constraints.write_text("\n".join(pins) + "\n")
        run([sys.executable, "-m", "pip", "install", "-c", constraints,
             "-r", WORKSPACE / "requirements.txt", "huggingface_hub"])
    after = torch_snapshot()
    require_a100(after)
    if (before["torch"], before["cuda"]) != (after["torch"], after["cuda"]):
        raise RuntimeError("The Torch/CUDA stack changed during dependency installation.")

    from huggingface_hub import hf_hub_download

    for name, size in MODEL_FILES.items():
        print(f"Download/verify BF16: {name}")
        path = Path(hf_hub_download(
            repo_id=MODEL_REPO, filename=name, revision=MODEL_REVISION,
            local_dir=WORKSPACE / "models",
        ))
        verify_model_file(path, size)

    if not shutil.which("cloudflared"):
        # Transport only; weights always come from the pinned HF revision above.
        run(["curl", "-fL", "--retry", "3", "-o", "/tmp/qwen21-bf16-cloudflared.deb",
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"])
        run(["dpkg", "-i", "/tmp/qwen21-bf16-cloudflared.deb"])

    manifest = {"profile": "qwen_image_2_1_bf16", "comfyui_commit": COMFYUI_COMMIT,
                "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "model_files": MODEL_FILES, "hardware": after}
    (WORKSPACE / "qwen21-bf16-profile.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Setup complete: {WORKSPACE}. Run this kit's 02_start_comfyui.py next.")


if __name__ == "__main__":
    setup()
