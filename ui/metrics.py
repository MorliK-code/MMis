"""UI utility helpers for lightweight metrics math."""

from ui.constants import FLOPS_PER_TOKEN


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def est_tflops(tokens_per_sec: float) -> float:
    return (FLOPS_PER_TOKEN * tokens_per_sec) / 1e12


def ms_to_s_text(ms_value: float) -> str:
    return f"{(float(ms_value) / 1000.0):.1f} с"

