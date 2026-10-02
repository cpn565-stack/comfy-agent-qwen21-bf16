"""Exercise preflight without installing dependencies or downloading weights."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen21_bf16_setup", ROOT / "scripts/colab/qwen_image_2_1_bf16/01_setup.py",
)
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class PreflightTests(unittest.TestCase):
    def hardware(self, **updates):
        info = dict(available=True, name="NVIDIA A100-SXM4-40GB", bf16=True,
                    vram_bytes=int(39.4935 * SETUP.GIB), torch="2.11.0+cu128", cuda="12.8")
        info.update(updates)
        return info

    def test_rejects_unsupported_assignments(self):
        for info in [self.hardware(available=False), self.hardware(name="NVIDIA L4"),
                     self.hardware(name="Tesla T4"), self.hardware(bf16=False),
                     self.hardware(vram_bytes=10 * SETUP.GIB)]:
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                SETUP.require_a100(info)

    def test_accepts_actual_a100_memory_and_80gb(self):
        SETUP.require_a100(self.hardware())
        SETUP.require_a100(self.hardware(name="NVIDIA A100 80GB PCIe", vram_bytes=80 * SETUP.GIB))

    def test_wrong_gpu_stops_before_checkout_or_install(self):
        with patch.object(SETUP.sys, "platform", "linux"), \
             patch.object(SETUP.Path, "is_dir", return_value=True), \
             patch.object(SETUP, "torch_snapshot", return_value=self.hardware(name="NVIDIA L4")), \
             patch.object(SETUP, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "A100 required"):
                SETUP.setup()
            run.assert_not_called()

    def test_resource_check_distinguishes_ram_and_disk(self):
        with self.assertRaisesRegex(RuntimeError, "system RAM"):
            SETUP.require_resources(12 * SETUP.GIB, 100 * SETUP.GIB, 32 * SETUP.GIB)
        with self.assertRaisesRegex(RuntimeError, "free disk"):
            SETUP.require_resources(64 * SETUP.GIB, 35 * SETUP.GIB, 32 * SETUP.GIB)
        SETUP.require_resources(64 * SETUP.GIB, 11 * SETUP.GIB, 0)

    def model_file(self, directory, dtype):
        header = json.dumps({"weight": {"dtype": dtype, "shape": [4],
                                        "data_offsets": [0, 8]}}).encode()
        path = Path(directory) / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
        return path

    def test_rejects_quantized_and_incomplete_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for dtype in ["I8", "U8", "F8_E4M3", "F16"]:
                path = self.model_file(directory, dtype)
                with self.subTest(dtype=dtype), self.assertRaises(RuntimeError):
                    SETUP.verify_model_file(path, path.stat().st_size)
            path = self.model_file(directory, "BF16")
            SETUP.verify_model_file(path, path.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                SETUP.verify_model_file(path, path.stat().st_size + 1)


if __name__ == "__main__":
    unittest.main()
