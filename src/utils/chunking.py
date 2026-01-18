from __future__ import annotations
from typing import List

def simple_chunk(text: str, max_chars: int = 1000) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in text.splitlines():
        add_len = len(line) + (1 if current else 0)  # +1 ~ newline między liniami

        if current and current_len + add_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(line)
        current_len += add_len

    if current:
        chunks.append("\n".join(current))

    return [c for c in chunks if c.strip()]