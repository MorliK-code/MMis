"""UI-specific constants and styles."""

CHAT_CSS = """
<style>
/* общие */
.msg { margin: 10px 0 18px 0; }
.name { font-weight: 600; margin-bottom: 6px; }
.bubble {
  display: inline-block;
  padding: 10px 12px;
  border-radius: 12px;
  max-width: 820px;
  line-height: 1.35;
  white-space: pre-wrap;
}

/* твои сообщения */
.msg.user .bubble { background: rgba(120,120,120,0.14); }

/* её сообщения */
.msg.ai .bubble { background: rgba(120,120,120,0.09); }

/* статистика под её ответом */
.stats {
  margin-top: 2px;
  display: block;
  font-size: 8px;
  line-height: 1.2;
  color: rgba(0,0,0,0.20);
}

/* если вдруг используешь тёмную тему — раскомментируй:
.stats { color: rgba(255,255,255,0.50); }
*/
.sep { height: 10px; }
</style>
"""

FLOPS_PER_TOKEN = 14e9
SHOW_TFLOPS_EST = True
