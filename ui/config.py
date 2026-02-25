"""UI-specific constants and styles."""

FLOPS_PER_TOKEN = 14e9
SHOW_TFLOPS_EST = True

DEFAULT_TEXT_SIZE = 11
DEFAULT_BUBBLE_OPACITY = 0.12
STATS_TEXT_SIZE_PX = 10
STATS_TEXT_ALPHA = 0.2


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_chat_css(font_size_px: int, bubble_opacity: float) -> str:
    font_size_px = int(_clamp(float(font_size_px), 9, 28))
    bubble_opacity = _clamp(float(bubble_opacity), 0.05, 0.9)
    user_opacity = _clamp(bubble_opacity + 0.06, 0.05, 0.95)

    return f"""
<style>
.msg {{ margin: 10px 0 18px 0; }}
.name {{ font-weight: 600; margin-bottom: 6px; font-size: {font_size_px}px; }}
.bubble {{
  display: inline-block;
  padding: 10px 12px;
  border-radius: 12px;
  max-width: 820px;
  line-height: 1.38;
  white-space: pre-wrap;
  font-size: {font_size_px}px;
}}
.msg.user .bubble {{ background: rgba(120,120,120,{user_opacity:.3f}); }}
.msg.ai .bubble {{ background: rgba(120,120,120,{bubble_opacity:.3f}); }}
.msg.ai .stats {{
  margin-top: 2px;
  display: block;
  font-size: {STATS_TEXT_SIZE_PX}px;
  line-height: 1.2;
  color: rgba(0,0,0,{STATS_TEXT_ALPHA:.3f});
}}
.sep {{ height: 10px; }}
</style>
"""
