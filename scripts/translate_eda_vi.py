"""Create a Vietnamese reader-facing copy of notebooks/01_eda.ipynb."""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "01_eda.ipynb"
TARGET = ROOT / "notebooks" / "01_eda_vi.ipynb"
CACHE = ROOT / "notebooks" / ".01_eda_vi_translation_cache.json"

VIETNAMESE = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]|[áàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]"
)
PROTECTED = re.compile(
    r"(<[^>]+>|https?://\S+|`[^`]+`|\b[A-Z][A-Z0-9_]{2,}\b|"
    r"\b[\w.-]+\.csv\b|\b\d+(?:\.\d+)?%?\b)"
)


def mostly_vietnamese(text: str) -> bool:
    return bool(VIETNAMESE.search(text))


def split_text(text: str, limit: int = 420) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= limit:
            current += part
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(part) > limit:
            cut = max(part.rfind(". ", 0, limit), part.rfind("\n", 0, limit))
            if cut < 80:
                cut = limit
            else:
                cut += 1
            chunks.append(part[:cut])
            part = part[cut:]
        current = part
    if current:
        chunks.append(current)
    return chunks


def protect(text: str) -> tuple[str, dict[str, str]]:
    values: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        key = f"ZXQ{len(values):04d}QXZ"
        values[key] = match.group(0)
        return key

    return PROTECTED.sub(repl, text), values


def restore(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(key, value).replace(key.lower(), value)
    return text


def translate_chunk(text: str) -> str:
    protected, values = protect(text)
    url = (
        "https://api.mymemory.translated.net/get?q="
        + quote(protected)
        + "&langpair=en%7Cvi&de=codex.translation%40example.com"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            translated = payload.get("responseData", {}).get("translatedText", "")
            if not translated or "QUERY LENGTH LIMIT" in translated.upper():
                raise RuntimeError(payload.get("responseDetails") or "Empty translation")
            return restore(html.unescape(translated), values)
        except Exception as exc:  # network services can be transient
            last_error = exc
            time.sleep(min(45, 3**attempt))
    raise RuntimeError(f"Translation failed: {last_error}")


def main() -> None:
    notebook = nbformat.read(SOURCE, as_version=4)
    cache: dict[str, str] = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}
    changed = 0

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown" or mostly_vietnamese(cell.source):
            continue
        translated_parts: list[str] = []
        for chunk in split_text(cell.source):
            if not re.search(r"[A-Za-z]{3}", chunk):
                translated_parts.append(chunk)
                continue
            if chunk not in cache:
                cache[chunk] = translate_chunk(chunk)
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
                time.sleep(1.5)
            translated_parts.append(cache[chunk])
        cell.source = "".join(translated_parts)
        changed += 1
        print(f"Translated markdown cell {index}", flush=True)

    nbformat.write(notebook, TARGET)
    print(f"Wrote {TARGET} ({changed} translated markdown cells)")


if __name__ == "__main__":
    main()
