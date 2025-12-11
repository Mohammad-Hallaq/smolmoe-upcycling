# src/moe_llm/utils.py
import time
from functools import wraps
from typing import Any, Callable, Dict

import torch


def timed(fn: Callable) -> Callable:
    """Decorator to print wall-clock time for a function call."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        out = fn(*args, **kwargs)
        total_time = time.perf_counter() - start_time
        print(f"time={total_time:.3f}s")
        return out
    return wrapper


def labelthis(label: str) -> Callable:
    """Attach a string label to a function (for logging/plotting)."""
    def deco(fn: Callable) -> Callable:
        fn.label = label
        return fn
    return deco


def pretty_dt(s: float) -> str:
    """Pretty formatting for durations in seconds."""
    if s < 1e-6:
        return f"{s * 1e9:.0f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 60:
        return f"{s:.3f} s"
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{int(m)}m {int(s)}s" if h < 1 else f"{int(h)}h {int(m)}m {int(s)}s")


def detach_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively move tensors to CPU and convert scalars to Python floats.
    Useful before logging/plotting metrics.
    """

    def to_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item() if x.dim() == 0 else x.detach().cpu().tolist()
        if isinstance(x, list):
            return [to_cpu(y) for y in x]
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        return x

    return {k: to_cpu(v) for k, v in metrics.items()}
