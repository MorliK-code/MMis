from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class OCRLine:
    text: str
    bbox: tuple[int, int, int, int]
    conf: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OCRExtractor:
    def __init__(self):
        self._tesseract_ready = self._detect_tesseract()

    def extract_text(self, image: str | Path | bytes) -> list[OCRLine]:
        if self._tesseract_ready:
            try:
                lines = self._extract_with_tesseract(image)
                if lines:
                    return self._postprocess(lines)
            except Exception as exc:
                LOGGER.warning("OCR tesseract path failed, using fallback: %s", exc)

        return self._postprocess(self._extract_fallback(image))

    def merge_lines(self, lines: list[OCRLine]) -> str:
        return "\n".join([x.text for x in lines if str(x.text).strip()]).strip()

    def find(self, lines: list[OCRLine], pattern: str, flags: int = re.IGNORECASE) -> tuple[int, int, int, int] | None:
        if not pattern:
            return None
        rx = re.compile(pattern, flags)
        for line in lines:
            if rx.search(line.text or ""):
                return line.bbox
        return None

    def _extract_with_tesseract(self, image: str | Path | bytes) -> list[OCRLine]:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        if isinstance(image, (str, Path)):
            obj = Image.open(str(image))
        elif isinstance(image, (bytes, bytearray)):
            import io

            obj = Image.open(io.BytesIO(bytes(image)))
        else:
            return []

        data = pytesseract.image_to_data(obj, output_type=pytesseract.Output.DICT)
        out: list[OCRLine] = []
        for i in range(len(data.get("text", []))):
            txt = str(data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data.get("conf", [0])[i])
            except Exception:
                conf = 0.0
            x = int(data.get("left", [0])[i] or 0)
            y = int(data.get("top", [0])[i] or 0)
            w = int(data.get("width", [0])[i] or 0)
            h = int(data.get("height", [0])[i] or 0)
            out.append(OCRLine(text=txt, bbox=(x, y, w, h), conf=max(0.0, min(1.0, conf / 100.0))))
        return out

    def _extract_fallback(self, image: str | Path | bytes) -> list[OCRLine]:
        text_blob = ""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                if path.suffix.lower() in {".txt", ".log", ".md"}:
                    text_blob = path.read_text(encoding="utf-8-sig", errors="ignore")
                else:
                    sidecar = path.with_suffix(".txt")
                    if sidecar.exists():
                        text_blob = sidecar.read_text(encoding="utf-8-sig", errors="ignore")
            else:
                text_blob = str(image)
        elif isinstance(image, (bytes, bytearray)):
            try:
                text_blob = bytes(image).decode("utf-8", errors="ignore")
            except Exception:
                text_blob = ""

        lines = [x.strip() for x in str(text_blob or "").splitlines() if x.strip()]
        out: list[OCRLine] = []
        for idx, line in enumerate(lines):
            out.append(OCRLine(text=line, bbox=(0, idx * 20, max(8, len(line) * 7), 18), conf=0.2))
        return out

    def _postprocess(self, lines: list[OCRLine]) -> list[OCRLine]:
        out: list[OCRLine] = []
        seen: set[str] = set()
        for line in lines:
            text = re.sub(r"\s+", " ", str(line.text or "")).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(OCRLine(text=text, bbox=tuple(line.bbox), conf=float(line.conf)))
        return out

    @staticmethod
    def _detect_tesseract() -> bool:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401

            return True
        except Exception:
            return False


_DEFAULT_OCR = OCRExtractor()


def extract_text(image: str | Path | bytes) -> list[OCRLine]:
    return _DEFAULT_OCR.extract_text(image)


def find(lines: list[OCRLine], pattern: str, flags: int = re.IGNORECASE) -> tuple[int, int, int, int] | None:
    return _DEFAULT_OCR.find(lines, pattern, flags)
