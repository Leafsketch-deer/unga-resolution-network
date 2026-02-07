#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Optional, Dict

from pypdf import PdfReader

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
PDF_DIR = DUMP_DIR / "pdf_en"
RECORDS_JSONL = DUMP_DIR / "records.jsonl"

SYMBOLS_TXT = pathlib.Path("ga_symbols_from_research_un") / "symbols.txt"

OUT_TSV = DUMP_DIR / "symbol_title.tsv"
OUT_JSONL = DUMP_DIR / "title_rebuild_log.jsonl"

DUMP_DIR.mkdir(parents=True, exist_ok=True)

# ========= regex / heuristics =========

# symbols: A/RES/70/1, A/RES/1514(XV), A/RES/77/1A ...
RE_SYM_LINE = re.compile(r"^A/RES/\S+$", re.IGNORECASE)

# PDF noise: page counters like "3/23"
RE_PAGE_COUNTER = re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$")

# "105th plenary meeting"
RE_PLENARY = re.compile(r"^\s*\d{1,3}(st|nd|rd|th)\s+plenary\s+meeting\b", re.IGNORECASE)

# Stop lines that indicate we’ve left the title region / moved into boilerplate
RE_STOP = re.compile(
    r"^\s*("
    r"The General Assembly\b|General Assembly\b|"
    r"Agenda item\b|Agenda items\b|"
    r"Distr\.\b|"
    r"A/RES/\b|"
    r"United Nations\b|"
    r"Seventy-.*session\b|"
    r"Sixty-.*session\b|"
    r"Fifty-.*session\b|"
    r"Forty-.*session\b|"
    r"Thirty-.*session\b|"
    r"Twenty-.*session\b|"
    r"Nineteen\b|E/RES/\b"
    r")",
    re.IGNORECASE,
)

# Headings on PDF:
# "70/1. Transforming our world: ..."
RE_HEAD_SLASH_LINE = re.compile(
    r"^\s*(\d{1,3})\s*/\s*(\d{1,4}[A-Z]?)\s*\.\s*(.+?)\s*$",
    re.IGNORECASE,
)

# "2625 (XXV). Declaration on ..."
RE_HEAD_NUM_ROMAN_LINE = re.compile(
    r"^\s*(\d{1,5})\s*\(\s*([IVXLCDM]+)\s*\)\s*\.\s*(.+?)\s*$",
    re.IGNORECASE,
)

# Cut phrases that frequently appear after title block
RE_CUT_PHRASES = re.compile(
    r"\b(Resolution adopted|Adopted at|Adopted on|The General Assembly)\b",
    re.IGNORECASE,
)

BAD_TITLES = {
    "",
    "document viewer",
    "undocs",
    "united nations",
    "united nations document system",
}

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def normalize_ws(s: str) -> str:
    # soft hyphen
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def looks_bad_title(t: str | None) -> bool:
    if not t:
        return True
    x = normalize_ws(t).lower()
    if x in BAD_TITLES:
        return True
    if "document viewer" in x:
        return True
    if len(x) < 4:
        return True
    # common false positives
    if re.search(r"\bplenary meeting\b", x):
        return True
    if re.match(r"^\d{1,3}(st|nd|rd|th)\b", x):
        return True
    return False

def load_symbols(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip().upper()
        if not s:
            continue
        if RE_SYM_LINE.match(s) and s.startswith("A/RES/"):
            out.append(s)
    # unique keeping order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def symbol_to_pdf_path(sym: str) -> pathlib.Path:
    # download naming rule: sym -> A_RES_70_1.pdf etc.
    stem = sym.replace("/", "_").replace("(", "").replace(")", "")
    return PDF_DIR / f"{stem}.pdf"

def append_jsonl(path: pathlib.Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _dehyphenate_lines(lines: list[str]) -> list[str]:
    """
    Join hyphenated line breaks: "im-" + "prison" => "imprison"
    """
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if cur.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()
            out.append((cur[:-1] + nxt).strip())
            i += 2
        else:
            out.append(cur)
            i += 1
    return out

def _clean_lines_from_pdf_text(text: str) -> list[str]:
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    raw_lines = [ln for ln in raw_lines if not RE_PAGE_COUNTER.match(ln)]
    raw_lines = _dehyphenate_lines(raw_lines)

    # keep as-is; don’t normalize away structure
    return raw_lines

def _is_noise_line(ln: str) -> bool:
    if RE_PLENARY.match(ln):
        return True
    return False

def extract_title_from_pdf(pdf_path: pathlib.Path) -> str | None:
    """
    Best-effort title extraction from EN PDF.
    Strategy:
    - Read first page text (and optionally second page if first fails)
    - Look for heading lines:
      - "70/1. <title>"
      - "2625 (XXV). <title>"
    - Concatenate a few following lines until STOP conditions.
    """
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        return None

    try:
        reader = PdfReader(str(pdf_path))
        if not reader.pages:
            return None
    except Exception:
        return None

    def try_pages(page_indexes: list[int]) -> Optional[str]:
        for pi in page_indexes:
            if pi < 0 or pi >= len(reader.pages):
                continue
            try:
                txt = reader.pages[pi].extract_text() or ""
            except Exception:
                continue
            if not txt.strip():
                continue

            lines = _clean_lines_from_pdf_text(txt)

            # 1) "70/1. Title"
            for i, ln in enumerate(lines):
                m = RE_HEAD_SLASH_LINE.match(ln)
                if not m:
                    continue

                title_parts = [m.group(3).strip()]
                for j in range(i + 1, min(i + 10, len(lines))):
                    nxt = lines[j]
                    if _is_noise_line(nxt):
                        continue
                    if RE_STOP.match(nxt):
                        break
                    if RE_HEAD_SLASH_LINE.match(nxt) or RE_HEAD_NUM_ROMAN_LINE.match(nxt):
                        break
                    title_parts.append(nxt)

                title = normalize_ws(" ".join(title_parts))
                title = RE_CUT_PHRASES.split(title, maxsplit=1)[0]
                title = normalize_ws(title).strip(" .;:-")

                if title and not looks_bad_title(title):
                    return title

            # 2) "2625 (XXV). Title"
            for i, ln in enumerate(lines):
                m = RE_HEAD_NUM_ROMAN_LINE.match(ln)
                if not m:
                    continue

                title_parts = [m.group(3).strip()]
                for j in range(i + 1, min(i + 10, len(lines))):
                    nxt = lines[j]
                    if _is_noise_line(nxt):
                        continue
                    if RE_STOP.match(nxt):
                        break
                    if RE_HEAD_SLASH_LINE.match(nxt) or RE_HEAD_NUM_ROMAN_LINE.match(nxt):
                        break
                    title_parts.append(nxt)

                title = normalize_ws(" ".join(title_parts))
                title = RE_CUT_PHRASES.split(title, maxsplit=1)[0]
                title = normalize_ws(title).strip(" .;:-")

                if title and not looks_bad_title(title):
                    return title

        return None

    # First try page 0; if it fails, try page 1 as fallback (some PDFs have cover/header on page 1)
    t = try_pages([0])
    if t:
        return t
    return try_pages([1])

def build_title_from_records_jsonl_map() -> Dict[str, str]:
    """
    Build a dict {SYMBOL: title-ish} from records.jsonl once.
    This is fallback only; may be noisy / generic for some records.
    """
    out: Dict[str, str] = {}
    if not RECORDS_JSONL.exists():
        return out

    with RECORDS_JSONL.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue

            sym = (obj.get("symbol") or "").strip().upper()
            if not sym.startswith("A/RES/"):
                continue

            rec = obj.get("record")
            if isinstance(rec, list) and rec:
                rec0 = rec[0]
            else:
                rec0 = rec

            if not isinstance(rec0, dict):
                continue

            # Candidate fields (depends on what recjson contains)
            cands: list[str] = []

            for key in ("title", "title_statement"):
                v = rec0.get(key)
                if isinstance(v, str) and v.strip():
                    cands.append(v.strip())

            v = rec0.get("titles")
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, str) and it.strip():
                        cands.append(it.strip())
                    elif isinstance(it, dict) and isinstance(it.get("value"), str):
                        cands.append(it["value"].strip())

            # pick first acceptable
            for cand in cands:
                t = normalize_ws(cand)
                if not looks_bad_title(t):
                    # keep first seen
                    out.setdefault(sym, t)
                    break

    return out

def main() -> None:
    syms = load_symbols(SYMBOLS_TXT)
    print("symbols:", len(syms))

    # Build fallback map once (fast)
    rec_title_map = build_title_from_records_jsonl_map()
    print("records.jsonl titles mapped:", len(rec_title_map))

    # overwrite outputs every run
    OUT_TSV.write_text("symbol\ttitle\tsource\tpdf_file\tts_utc\n", encoding="utf-8")
    OUT_JSONL.write_text("", encoding="utf-8")

    ok_pdf = 0
    ok_rec = 0
    miss = 0

    for i, sym in enumerate(syms, 1):
        pdf_path = symbol_to_pdf_path(sym)
        title = None
        source = None

        # A) PDF-first
        t1 = extract_title_from_pdf(pdf_path)
        if t1 and not looks_bad_title(t1):
            title = t1
            source = "pdf_heading"

        # B) fallback: records.jsonl map
        if not title:
            t2 = rec_title_map.get(sym)
            if t2 and not looks_bad_title(t2):
                title = t2
                source = "undl_recjson"

        if not title:
            title = ""
            source = "missing"
            miss += 1
        else:
            if source == "pdf_heading":
                ok_pdf += 1
            elif source == "undl_recjson":
                ok_rec += 1

        ts = now_iso()
        pdf_name = pdf_path.name if pdf_path.exists() else ""

        with OUT_TSV.open("a", encoding="utf-8") as out:
            out.write(f"{sym}\t{title}\t{source}\t{pdf_name}\t{ts}\n")

        append_jsonl(OUT_JSONL, {
            "ts_utc": ts,
            "symbol": sym,
            "title": title,
            "source": source,
            "pdf": str(pdf_path) if pdf_path.exists() else None,
        })

        if i <= 10 or i % 500 == 0:
            show = title[:90]
            print(f"[{i}/{len(syms)}] {sym} source={source} title={show}")

    print("done")
    print("total:", len(syms))
    print("titled(pdf):", ok_pdf, "titled(recjson):", ok_rec, "missing:", miss)
    print("wrote:", OUT_TSV)
    print("log :", OUT_JSONL)

if __name__ == "__main__":
    main()