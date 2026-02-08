#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pypdf import PdfReader

# =========================
# Paths
# =========================
DUMP_DIR = pathlib.Path("undl_dump")
PDF_DIR = DUMP_DIR / "pdf_en"

OUT_CSV = DUMP_DIR / "edges.csv"
OUT_SUMMARY = DUMP_DIR / "edges_summary.json"

OUT_REL_SUMMARY_TSV = DUMP_DIR / "clause_relation_summary.tsv"
OUT_REL_SUMMARY_JSON = DUMP_DIR / "clause_relation_summary.json"

# =========================
# Symbol extraction regexes
# =========================

RE_A_RES = re.compile(r"\bA/RES/\d+(?:/\d+|\([IVXLCDM]+\))\b", re.IGNORECASE)
RE_RES_NUM_ROMAN = re.compile(
    r"\bresolution\s+(\d{1,5})\s*\(\s*([IVXLCDM]+)\s*\)", re.IGNORECASE
)
RE_RES_SLASH = re.compile(
    r"\bresolution\s+(\d{1,3})\s*/\s*(\d{1,4})\b", re.IGNORECASE
)
RE_GA_RES_SLASH = re.compile(
    r"\bgeneral\s+assembly\s+resolution\s+(\d{1,3})\s*/\s*(\d{1,4})\b",
    re.IGNORECASE,
)

RE_RES_LIST_HEAD = re.compile(r"\b(?:its\s+)?resolutions?\b", re.IGNORECASE)
RE_SESS_SLASH_OF = re.compile(
    r"\b(\d{1,3})\s*/\s*(\d{1,4})\s+of\b", re.IGNORECASE
)

# =========================
# PDF text extraction + cleaning
# =========================

RE_PAGE_COUNTER_LINE = re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$")
RE_SOFT_HYPHEN = re.compile(r"\u00ad")

def extract_text_pypdf(pdf_path: pathlib.Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            parts.append(t)
    return "\n".join(parts)

def clean_lines(raw_text: str) -> List[str]:
    raw_text = RE_SOFT_HYPHEN.sub("", raw_text)
    lines = [x.strip() for x in raw_text.splitlines()]
    lines = [ln for ln in lines if ln]
    lines = [ln for ln in lines if not RE_PAGE_COUNTER_LINE.match(ln)]
    return lines

# =========================
# Clause starters
# =========================

CLAUSE_STARTERS = [
    r"recall(?:s|ing|ed)?",
    r"reaffirm(?:s|ing|ed)?",
    r"affirm(?:s|ing|ed)?",
    r"reiterate(?:s|ing|d)?",
    r"stress(?:es|ing|ed)?",
    r"emphasiz(?:ing|es|ed)?",
    r"welcom(?:ing|es|ed)?",
    r"recogniz(?:ing|es|ed)?",
    r"acknowledg(?:ing|es|ed)?",
    r"concerned",
    r"noting",
    r"taking\s+note",
    r"bearing\s+in\s+mind",
    r"mindful",
    r"having\s+considered",
    r"guided\s+by",
]

RE_ANY_STARTER = re.compile(
    r"\b(" + "|".join(CLAUSE_STARTERS) + r")\b", re.IGNORECASE
)

RE_EDGE_STARTER = RE_ANY_STARTER  # edge抽出対象は全starter

def edge_rel_base(word: str) -> Optional[str]:
    w = (word or "").strip().lower()
    w = re.sub(r"\s+", " ", w)

    if w.startswith("taking note"):
        return "taking note"
    if w.startswith("bearing in mind"):
        return "bearing in mind"
    if w.startswith("having considered"):
        return "having considered"
    if w.startswith("guided by"):
        return "guided by"

    if w.startswith("recall"):
        return "recall"
    if w.startswith("noting"):
        return "noting"
    if w.startswith("reaffirm"):
        return "reaffirm"
    if w.startswith("affirm"):
        return "affirm"
    if w.startswith("reiterate"):
        return "reiterate"
    if w.startswith("stress"):
        return "stress"
    if w.startswith("emphasiz"):
        return "emphasize"
    if w.startswith("welcom"):
        return "welcome"
    if w.startswith("recogniz"):
        return "recognize"
    if w.startswith("acknowledg"):
        return "acknowledge"
    if w.startswith("mindful"):
        return "mindful"
    if w.startswith("concerned"):
        return "concerned"

    return None

# =========================
# Clause extraction
# =========================

def extract_all_clauses(full_text: str) -> List[Tuple[str, str]]:
    text = re.sub(r"[ \t]+", " ", full_text)
    text = re.sub(r"\n{2,}", "\n", text)

    starters = [(m.start(), m.group(1)) for m in RE_ANY_STARTER.finditer(text)]
    if not starters:
        return []

    out: List[Tuple[str, str]] = []
    for i, (pos, w) in enumerate(starters):
        end = starters[i + 1][0] if i + 1 < len(starters) else len(text)
        chunk = text[pos:end].strip()
        if chunk:
            out.append((chunk, edge_rel_base(w) or w.lower()))
    return out

def extract_relation_chunks(full_text: str) -> List[Tuple[str, str]]:
    text = re.sub(r"[ \t]+", " ", full_text)
    text = re.sub(r"\n{2,}", "\n", text)

    starters = [(m.start(), m.group(1)) for m in RE_ANY_STARTER.finditer(text)]
    if not starters:
        return []

    out: List[Tuple[str, str]] = []
    for i, (pos, w) in enumerate(starters):
        if not RE_EDGE_STARTER.match(w):
            continue

        rel = edge_rel_base(w)
        if not rel:
            continue

        end = starters[i + 1][0] if i + 1 < len(starters) else len(text)
        chunk = text[pos:end].strip()
        if chunk:
            out.append((chunk, rel))
    return out

# =========================
# Citation symbol extraction
# =========================

def normalize_symbol(sym: str) -> str:
    sym = sym.strip().upper()
    sym = re.sub(r"\s+", "", sym)
    return sym

def cited_symbols(text: str) -> List[str]:
    out: Set[str] = set()

    for m in RE_A_RES.finditer(text):
        out.add(normalize_symbol(m.group(0)))

    for m in RE_RES_NUM_ROMAN.finditer(text):
        out.add(f"A/RES/{m.group(1)}({m.group(2).upper()})")

    for m in RE_GA_RES_SLASH.finditer(text):
        out.add(f"A/RES/{m.group(1)}/{m.group(2)}")
    for m in RE_RES_SLASH.finditer(text):
        out.add(f"A/RES/{m.group(1)}/{m.group(2)}")

    if RE_RES_LIST_HEAD.search(text):
        for m in RE_SESS_SLASH_OF.finditer(text):
            out.add(f"A/RES/{m.group(1)}/{m.group(2)}")

    return sorted(out)

def snippet(s: str, n: int = 260) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"

# =========================
# Main
# =========================

def main() -> None:
    files = sorted(PDF_DIR.glob("*.pdf"))
    if not files:
        raise FileNotFoundError(PDF_DIR)

    seen_edges: Set[Tuple[str, str, str]] = set()
    edges_written = 0

    clause_counter = Counter()
    clause_with_cite = Counter()
    clause_total_edges = Counter()

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "citing_symbol",
            "cited_symbol",
            "relation",
            "evidence_snippet",
            "pdf_file",
        ])

        for pdf in files:
            citing_symbol = pdf.stem.replace("_", "/").upper()

            try:
                raw = extract_text_pypdf(pdf)
            except Exception:
                continue

            lines = clean_lines(raw)
            text = "\n".join(lines)

            clauses = extract_all_clauses(text)
            for chunk, starter in clauses:
                clause_counter[starter] += 1
                cited = cited_symbols(chunk)
                if cited:
                    clause_with_cite[starter] += 1
                    clause_total_edges[starter] += len(cited)

            rel_chunks = extract_relation_chunks(text)
            for chunk, rel in rel_chunks:
                cited = cited_symbols(chunk)
                for c in cited:
                    if c == citing_symbol:
                        continue
                    key = (citing_symbol, c, rel)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    w.writerow([
                        citing_symbol,
                        c,
                        rel,
                        snippet(chunk),
                        pdf.name,
                    ])
                    edges_written += 1

    with OUT_REL_SUMMARY_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "clause_starter",
            "num_clauses",
            "clauses_with_citations",
            "total_cited_edges",
            "citation_ratio",
        ])
        for k in sorted(clause_counter):
            total = clause_counter[k]
            cited_n = clause_with_cite.get(k, 0)
            edges_n = clause_total_edges.get(k, 0)
            ratio = cited_n / total if total else 0.0
            w.writerow([k, total, cited_n, edges_n, f"{ratio:.4f}"])

    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "pdf_count": len(files),
                "edges_written": edges_written,
                "method": "extract edges from VALUE_NEUTRAL + STANCE clause starters",
                "clause_distribution_tsv": str(OUT_REL_SUMMARY_TSV),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("done")
    print("edges written:", edges_written)

if __name__ == "__main__":
    main()