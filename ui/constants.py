"""UI constants and presets."""

from ui import config as ui_config

DEFAULT_TEXT_SIZE = getattr(ui_config, "DEFAULT_TEXT_SIZE", 14)
DEFAULT_BUBBLE_OPACITY = getattr(ui_config, "DEFAULT_BUBBLE_OPACITY", 0.1)
FLOPS_PER_TOKEN = getattr(ui_config, "FLOPS_PER_TOKEN", 14e9)
SHOW_TFLOPS_EST = getattr(ui_config, "SHOW_TFLOPS_EST", True)

FEMALE_TONE_PRESETS = [
    ("Мягкий RU", "ru-RU-SvetlanaNeural"),
    ("Нейтральный EN", "en-US-JennyNeural"),
    ("Теплый EN", "en-GB-SoniaNeural"),
    ("Энергичный DE", "de-DE-KatjaNeural"),
    ("Спокойный FR", "fr-FR-DeniseNeural"),
]

