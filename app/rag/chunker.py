"""
Heading-aware paragraph chunker for Sharia finance Markdown documents.

Strategy
--------
1. Walk lines and detect ATX headings (# / ## / ###).
2. Each heading boundary starts a new *section*; the section inherits its full
   ancestor path (e.g. "Murabaha Overview > Key Sharia Conditions").
3. Within a section, paragraphs (separated by blank lines) are accumulated until
   they would exceed `max_words`. At that point the accumulated block is emitted
   as a chunk and a new block begins.
4. Numbered lists (lines that start "1." "2." etc.) are kept together inside
   whichever paragraph block they fall in — they are NOT split on blank lines
   within the list (heuristic: consecutive lines matching r"^\d+\." are merged).
5. Every chunk carries its heading_path as a prefix so the LLM can see context.

Result
------
Each chunk looks like:

    [Murabaha Overview > Key Sharia Conditions]

    1. **Asset must exist**: The financier must first own the asset before
       selling it. Selling something not yet owned (bay al-ma'dum) is prohibited.
    2. **Disclosure of cost**: The original purchase price and the profit margin
       must be clearly disclosed to the buyer.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    heading: str
    page_number: int = 1


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")
_NUMBERED_RE = re.compile(r"^\s*\d+[\.\)]\s")


def _heading_level_and_title(line: str) -> Optional[tuple[int, str]]:
    m = _HEADING_RE.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def _merge_list_items(paragraphs: list[str]) -> list[str]:
    """
    Merge consecutive paragraphs that look like numbered-list continuations
    so that "1. ...\n\n2. ..." stays as one logical block.
    """
    merged: list[str] = []
    for para in paragraphs:
        if (
            merged
            and _NUMBERED_RE.match(para)
            and _NUMBERED_RE.match(merged[-1].split("\n")[-1])
        ):
            merged[-1] = merged[-1] + "\n" + para
        else:
            merged.append(para)
    return merged


def chunk_markdown(
    text: str,
    source: str,
    start_id: int = 0,
    max_words: int = 400,
) -> list[Chunk]:
    """
    Parse *text* (Markdown) and return a list of Chunk objects.

    Parameters
    ----------
    text      : raw Markdown string
    source    : document identifier (e.g. filename stem)
    start_id  : integer ID offset for Qdrant point IDs
    max_words : soft upper-bound words per chunk
    """
    lines = text.splitlines()

    # ── Phase 1: segment into (heading_path, raw_lines_with_page) pairs ──────
    sections: list[tuple[str, list[tuple[str, int]]]] = []
    heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    current_lines: list[tuple[str, int]] = []
    current_page = 1

    def _flush_section() -> None:
        path = " > ".join(title for _, title in heading_stack)
        sections.append((path, current_lines[:]))
        current_lines.clear()

    page_break_re = re.compile(r"<!--\s*PAGE_BREAK\s+page=\"(\d+)\"\s*-->")

    for line in lines:
        pb_match = page_break_re.match(line.strip())
        if pb_match:
            current_page = int(pb_match.group(1))
            continue

        parsed = _heading_level_and_title(line)
        if parsed:
            level, title = parsed
            _flush_section()
            # Pop stack entries at same or deeper level
            heading_stack = [(l, t) for l, t in heading_stack if l < level]
            heading_stack.append((level, title))
        else:
            current_lines.append((line, current_page))

    _flush_section()  # trailing content

    # ── Phase 2: paragraph-split each section, group into chunks ─────────────
    chunks: list[Chunk] = []
    chunk_idx = 0

    for heading_path, raw_lines_with_page in sections:
        if not raw_lines_with_page:
            continue

        # Split on blank lines -> paragraphs (with pages tracked)
        paragraphs_with_page: list[tuple[str, int]] = []
        current_para_lines: list[str] = []
        para_page = None

        for line, page in raw_lines_with_page:
            if not line.strip():
                if current_para_lines:
                    paragraphs_with_page.append(
                        ("\n".join(current_para_lines).strip(), para_page or page)
                    )
                    current_para_lines.clear()
                    para_page = None
            else:
                if para_page is None:
                    para_page = page
                current_para_lines.append(line)
        if current_para_lines:
            paragraphs_with_page.append(
                ("\n".join(current_para_lines).strip(), para_page or current_page)
            )

        paragraphs_with_page = [(p, pg) for p, pg in paragraphs_with_page if p]

        # Merge list items
        merged_paragraphs: list[tuple[str, int]] = []
        for para, pg in paragraphs_with_page:
            if (
                merged_paragraphs
                and _NUMBERED_RE.match(para)
                and _NUMBERED_RE.match(merged_paragraphs[-1][0].split("\n")[-1])
            ):
                merged_paragraphs[-1] = (
                    merged_paragraphs[-1][0] + "\n" + para,
                    merged_paragraphs[-1][1],
                )
            else:
                merged_paragraphs.append((para, pg))

        # Accumulate paragraphs into chunks ≤ max_words
        bucket: list[str] = []
        chunk_page = 1
        word_count = 0

        def _emit_chunk() -> None:
            nonlocal chunk_idx
            if not bucket:
                return
            body = "\n\n".join(bucket)
            prefix = f"[{heading_path}]\n\n" if heading_path else ""
            chunks.append(
                Chunk(
                    id=start_id + chunk_idx,
                    text=prefix + body,
                    source=source,
                    heading=heading_path,
                    page_number=chunk_page,
                )
            )
            chunk_idx += 1

        for para, pg in merged_paragraphs:
            words = len(para.split())
            if bucket and word_count + words > max_words:
                _emit_chunk()
                bucket = [para]
                chunk_page = pg
                word_count = words
            else:
                if not bucket:
                    chunk_page = pg
                bucket.append(para)
                word_count += words

        _emit_chunk()

    return chunks
