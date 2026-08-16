"""Best-effort machine and environment capture for Phase-0 artifacts.

Every probe is guarded: a missing tool or library degrades to ``None`` instead
of failing the run. psutil/torch are imported lazily so this module stays
importable in CPU-only CI environments.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = ("torch", "transformers", "huggingface-hub", "psutil")


def capture_environment(git_root: Path | None = None) -> dict[str, Any]:
    """Collect OS/CPU/GPU/package/git identity for the current process."""

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "os": _os_manifest(),
        "cpu": _cpu_manifest(),
        "system_ram": _system_ram_manifest(),
        "gpu": _gpu_manifest(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        "git": _git_manifest(git_root),
    }


def write_environment_json(path: str | Path, environment: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(environment, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _os_manifest() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def _cpu_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "python_processor": platform.processor() or None,
    }
    try:
        import psutil

        frequencies = psutil.cpu_freq()
        manifest.update(
            {
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "max_frequency_mhz": frequencies.max if frequencies else None,
            }
        )
    except Exception:  # noqa: BLE001 - environment probes must never fail the run
        manifest["psutil_error"] = "unavailable"
    return manifest


def _system_ram_manifest() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_bytes": memory.total,
            "available_bytes_at_capture": memory.available,
        }
    except Exception:  # noqa: BLE001
        return {"total_bytes": None, "available_bytes_at_capture": None}


def _gpu_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {"cuda_available": None, "torch_cuda_version": None}
    try:
        import torch

        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            manifest["name"] = properties.name
            manifest["compute_capability"] = f"{properties.major}.{properties.minor}"
            manifest["total_vram_bytes"] = int(properties.total_memory)
    except Exception:  # noqa: BLE001
        manifest["torch"] = "unavailable"
    manifest["driver_version"] = _nvidia_smi_field("driver_version")
    return manifest


def _nvidia_smi_field(field: str) -> str | None:
    query = {
        "driver_version": "driver_version",
        "memory_used_mib": "memory.used",
        "memory_free_mib": "memory.free",
        "memory_total_mib": "memory.total",
    }.get(field)
    if query is None:
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return output.splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return None


def nvidia_smi_memory_mib() -> dict[str, int | None] | None:
    """Device-wide VRAM picture outside PyTorch's allocator, or None."""

    fields = {
        "memory_used_mib": "memory.used",
        "memory_free_mib": "memory.free",
        "memory_total_mib": "memory.total",
    }
    values: dict[str, int | None] = {}
    for key, query in fields.items():
        raw = _nvidia_smi_field_by_query(query)
        if raw is None:
            return None
        values[key] = int(raw)
    return values


def _nvidia_smi_field_by_query(query: str) -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return output.splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_manifest(git_root: Path | None) -> dict[str, Any]:
    if git_root is None:
        return {"branch": None, "commit": None, "working_tree_dirty": None}
    return {
        "branch": _git_query(git_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git_query(git_root, ["rev-parse", "HEAD"]),
        "working_tree_dirty": _git_dirty(git_root),
    }


def _git_query(root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _git_dirty(root: Path) -> bool | None:
    status = _git_query(root, ["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)
