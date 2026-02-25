"""UI-specific constants and styles."""

FLOPS_PER_TOKEN = 14e9
SHOW_TFLOPS_EST = True

DEFAULT_TEXT_SIZE = 14
DEFAULT_BUBBLE_OPACITY = 0.12


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_chat_css(font_size_px: int, bubble_opacity: float) -> str:
    font_size_px = int(_clamp(float(font_size_px), 9, 28))
    bubble_opacity = _clamp(float(bubble_opacity), 0.04, 0.3)
    user_opacity = _clamp(bubble_opacity + 0.03, 0.04, 0.35)

    return f"""
<style>

.msg {{
  margin: 10px 0 18px 0;
  padding: 8px 10px;
  border-radius: 10px;
}}

.name {{
  font-weight: 400;
  margin-bottom: 4px;
  font-size: 12px;
  color: rgba(255,255,255,0.72);
  background: transparent;
}}

.bubble {{
  display: block;
  padding: 0;
  border-radius: 0;
  max-width: 820px;
  line-height: 1.38;
  white-space: pre-wrap;
  font-size: {font_size_px}px;
  background: transparent;
}}

.msg.user {{ background: rgba(120,120,120,{user_opacity:.3f}); }}
.msg.ai {{ background: rgba(120,120,120,{bubble_opacity:.3f}); }}

.stats {{
  margin-top: 6px;
  padding: 0;
  background: transparent;
  display: block;
  font-size: 11px;
  line-height: 1.2;
  color: rgba(255,255,255,0.6);
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}}

.sep {{ height: 10px; }}
</style>

"""