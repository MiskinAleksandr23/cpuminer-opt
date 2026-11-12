"""Read a completed AVX-512 report for a descriptive, separate-run comparison."""
import json
import math
from pathlib import Path

AVX_LABELS = ("base", "rounds", "ternary", "combined")


def read_reference(directory, model, cpu, seconds):
    meta = json.loads((directory/"run.json").read_text())
    if not isinstance(meta, dict) or meta.get("status") != "complete" or meta.get("mode", "avx512") != "avx512":
        raise ValueError("expected a completed AVX-512 run")
    if (meta.get("cpu_model"), meta.get("cpu"), meta.get("measured_seconds")) != (model, cpu, seconds):
        raise ValueError("CPU model, logical CPU or measurement duration differs")
    report = json.loads((directory/"summary.json").read_text())
    rates = {label: float(report["variants"][label]["geometric_mean_h_s"]) for label in AVX_LABELS}
    if not all(math.isfinite(value) and value > 0 for value in rates.values()):
        raise ValueError("invalid reference rates")
    return {"directory": str(directory.resolve()), "started_utc": meta.get("started_utc"),
            "geometric_mean_h_s": rates}


def find_reference(parent, explicit, model, cpu, seconds):
    if explicit:
        directory = Path(explicit).expanduser().resolve()
        if directory.is_file() and directory.name == "summary.json":
            directory = directory.parent
        try:
            return read_reference(directory, model, cpu, seconds)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot use {directory} as an AVX-512 reference: {exc}") from exc
    candidates = sorted(parent.glob("*/summary.json"), key=lambda p: p.stat().st_mtime_ns,
                        reverse=True)
    for summary in candidates:
        try:
            return read_reference(summary.parent, model, cpu, seconds)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None
