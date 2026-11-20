"""Load an application-local Windows C++ runtime before native ML imports."""
import ctypes
import hashlib
import os
from pathlib import Path

MSVC_VERSION = "14.51.36231"
_handles = []


def bootstrap_native_runtime():
    """Prefer the pinned local MSVC runtime without changing system DLLs.

    Call before importing onnx, onnxruntime, or torch. A missing local runtime
    leaves native dependency resolution to the host, for Linux and existing
    Windows installations with a compatible C++ runtime.
    """
    if os.name != "nt":
        return {"source": "host", "platform": os.name}
    root = Path(__file__).resolve().parents[2]
    directory = (root / ".runtime/msvc" /
                 f"msvc_runtime-{MSVC_VERSION}.data/data/Scripts")
    dll = directory / "msvcp140.dll"
    if not dll.is_file():
        return {"source": "host", "setup_command": "python tools/setup_runtime.py"}
    if not _handles:
        directory_handle = os.add_dll_directory(str(directory))
        try:
            library = ctypes.WinDLL(str(dll))
        except OSError:
            directory_handle.close()
            raise
        _handles.extend([directory_handle, library])
    return {
        "source": "application_local", "version": MSVC_VERSION,
        "dll": str(dll), "sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
    }
