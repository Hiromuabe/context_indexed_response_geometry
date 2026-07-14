"""Print a plain-text environment record suitable for a release archive."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"


print(f"python={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")
for package in ("numpy", "torch", "transformers", "datasets", "matplotlib"):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package}={version}")
try:
    import torch

    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.backends.cudnn.version={torch.backends.cudnn.version()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"torch.cuda.device.{index}="
            f"{properties.name}; total_memory={properties.total_memory}"
        )
except (ImportError, RuntimeError) as exc:
    print(f"torch.cuda.details=unavailable ({exc})")
print("nvidia-smi:")
print(command("nvidia-smi"))
