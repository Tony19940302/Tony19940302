"""Download and unpack the pinned app-local Windows MSVC runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from battletracker.runtime import MSVC_VERSION, bootstrap_native_runtime


def main():
    if os.name != "nt":
        print(json.dumps(bootstrap_native_runtime()))
        return
    root = Path(__file__).resolve().parents[1]
    destination = root / ".runtime"
    destination.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(destination),
        f"msvc-runtime=={MSVC_VERSION}",
    ], check=True)
    wheels = list(destination.glob(f"msvc_runtime-{MSVC_VERSION}-*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one runtime wheel for this environment")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination / "msvc")
    evidence = dict(bootstrap_native_runtime(), wheel=wheel.name,
                    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    (destination / "runtime-manifest.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
