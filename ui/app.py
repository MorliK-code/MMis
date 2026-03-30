from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent

    def _norm_path(value: str) -> str:
        return os.path.normcase(os.path.abspath(str(value)))

    if all(_norm_path(path) != _norm_path(str(_ROOT)) for path in sys.path):
        sys.path.insert(0, str(_ROOT))
    sys.path = [path for path in sys.path if _norm_path(path) != _norm_path(str(_HERE))]

if sys.platform.startswith("win"):
    os.environ.setdefault("QT_FONT_DPI", "96")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "RoundPreferFloor")
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
    except Exception:
        pass

from PySide6.QtGui import QFont
from ui.chat_window import ChatWindow


def _main() -> int:
    from PySide6.QtWidgets import QApplication, QToolTip

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    font = QFont("Segoe UI")
    font.setWeight(QFont.Weight.Medium)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    app.setFont(font)
    tooltip_font = QFont("Segoe UI")
    tooltip_font.setWeight(QFont.Weight.Normal)
    tooltip_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    tooltip_font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    QToolTip.setFont(tooltip_font)
    win = ChatWindow()
    win.show()
    return app.exec()

MainWindow = ChatWindow
main = _main

__all__ = ["ChatWindow", "MainWindow", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
