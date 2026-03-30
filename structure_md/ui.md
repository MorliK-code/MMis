# ui.md

- project_root: `C:\Users\Morli.K\Desktop\MMis`
- note: old desktop UI branch removed; this file now lists the current active UI surface

| file | purpose |
|---|---|
| `ui/app.py` | current UI entrypoint, re-exporting the active chat window |
| `ui/chat_window.py` | main chat window logic |
| `ui/chat_shell.py` | visual shell/widgets base used by the current window |
| `ui/api_client.py` | API bridge for UI requests |
| `ui/workers.py` | streaming reply worker |
| `ui/chat_sessions.py` | persisted chat sessions/history |
| `ui/voice_adapter.py` | STT/TTS adapters |
| `ui/widgets/memory_inspector_panel.py` | memory inspector panel used by the chat window |
| `ui/style.css` | preserved for now as requested |
