#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pypdf import PdfReader

# =========================
# Paths
# =========================
DUMP_DIR = pathlib.Path("undl_dump")
PDF_DIR = DUMP_DIR / "pdf_en"

# edges output (same as before)
OUT_CSV = DUMP_DIR / "edges.csv"
OUT_SUMMARY = DUMP_DIR / "edges_summary.json"

# NEW: clause relation distribution (to judge whether recall/reaffirm/reiterate dominate)
OUT_REL_SUMMARY_TSV = DUMP_DIR / "clause_relation_summary.tsv"
OUT_REL_SUMMARY_JSON = DUMP_DIR / "clause_relation_summary.json"

# =========================
# Symbol extraction regexes
# =========================

# Direct A/RES/... format
RE_A_RES = re.compile(r"\bA/RES/\d+(?:/\d+|\([IVXLCDM]+\))\b", re.IGNORECASE)

# "resolution 2625 (XXV)" -> A/RES/2625(XXV)
RE_RES_NUM_ROMAN = re.compile(r"\bresolution\s+(\d{1,5})\s*\(\s*([IVXLCDM]+)\s*\)", re.IGNORECASE)

# "resolution 70/1" -> A/RES/70/1
RE_RES_SLASH = re.compile(r"\bresolution\s+(\d{1,3})\s*/\s*(\d{1,4})\b", re.IGNORECASE)

# "General Assembly resolution 70/1"
RE_GA_RES_SLASH = re.compile(r"\bgeneral\s+assembly\s+resolution\s+(\d{1,3})\s*/\s*(\d{1,4})\b", re.IGNORECASE)

# For list patterns like: "its resolutions 68/239 ... 69/226 ... and 70/210 ..."
RE_RES_LIST_HEAD = re.compile(r"\b(?:its\s+)?resolutions?\b", re.IGNORECASE)

# Capture "68/239" tokens, but avoid page counters like "3/23" by requiring context.
# We implement context-based matcher below.
RE_SESS_SLASH_TOKEN = re.compile(r"\b(\d{1,3})\s*/\s*(\d{1,4})\b")

# Special: token followed by "of" is very common in citations: "69/226 of 19 December 2014"
RE_SESS_SLASH_OF = re.compile(r"\b(\d{1,3})\s*/\s*(\d{1,4})\s+of\b", re.IGNORECASE)

# =========================
# PDF text extraction + cleaning
# =========================
RE_PAGE_COUNTER_LINE = re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$")  # lines that are exactly like "3/23"
RE_SOFT_HYPHEN = re.compile(r"\u00ad")  # soft hyphen

def extract_text_pypdf(pdf_path: pathlib.Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            parts.append(t)
    return "\n".join(parts)

def clean_lines(raw_text: str) -> List[str]:
    # remove soft hyphen
    raw_text = RE_SOFT_HYPHEN.sub("", raw_text)

    lines = [x.strip() for x in raw_text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty
    lines = [ln for ln in lines if not RE_PAGE_COUNTER_LINE.match(ln)]  # drop pure "3/23" lines
    return lines

# =========================
# Clause chunking (key change)
# =========================

# Clause starter vocabulary (broad, to cut chunks reasonably)
# All clauses starting with these should be considered for "relation distribution"
CLAUSE_STARTERS = [
    r"recall(?:s|ing|ed)?",
    r"reaffirm(?:s|ing|ed)?",
    r"reiterate(?:s|ing|d)?",

    # non-relation clause starters (boundary too, and also included in distribution)
    r"noting",
    r"taking\s+note",
    r"recogniz(?:ing|es|ed)?",
    r"bearing\s+in\s+mind",
    r"welcom(?:ing|es|ed)?",
    r"acknowledg(?:ing|es|ed)?",
    r"concerned",
    r"emphasiz(?:ing|es|ed)?",
    r"mindful",
    r"stress(?:ing|es|ed)?",
    r"affirm(?:ing|s|ed)?",
    r"guided\s+by",
    r"having\s+considered",
]

# Any starter for boundary and distribution
RE_ANY_STARTER = re.compile(r"\b(" + "|".join(CLAUSE_STARTERS) + r")\b", re.IGNORECASE)

# The three relation starters we focus on for edge extraction
RE_REL_STARTER = re.compile(
    r"\b(recalls?|recalling|recalled|reaffirms?|reaffirming|reaffirmed|reiterates?|reiterating|reiterated)\b",
    re.IGNORECASE
)

def rel_base(word: str) -> Optional[str]:
    w = (word or "").lower()
    if w.startswith("recall"):
        return "recalls"
    if w.startswith("reaffirm"):
        return "reaffirms"
    if w.startswith("reiterate"):
        return "reiterates"
    return None

def starter_base(word: str) -> str:
    """
    Normalize a starter token to a canonical form for distribution.
    We keep it simple: lowercase, compress whitespace.
    """
    w = (word or "").strip().lower()
    w = re.sub(r"\s+", " ", w)
    return w

def extract_all_clauses(full_text: str) -> List[Tuple[str, str]]:
    """
    Extract all clauses starting from ANY starter in CLAUSE_STARTERS.

    Returns:
      [(chunk_text, starter_base), ...]
    chunk = from starter position to just before next starter.
    """
    text = re.sub(r"[ \t]+", " ", full_text)
    text = re.sub(r"\n{2,}", "\n", text)

    starters = [(m.start(), m.group(1)) for m in RE_ANY_STARTER.finditer(text)]
    if not starters:
        return []

    clauses: List[Tuple[str, str]] = []
    for i, (pos, w) in enumerate(starters):
        end = starters[i + 1][0] if i + 1 < len(starters) else len(text)
        chunk = text[pos:end].strip()
        if not chunk:
            continue
        clauses.append((chunk, starter_base(w)))
    return clauses

def extract_relation_chunks(full_text: str) -> List[Tuple[str, str]]:
    """
    Extract only relation clauses (recalls/reaffirms/reiterates) as chunks.
    Returns:
      [(chunk_text, relation_base), ...]
    """
    text = re.sub(r"[ \t]+", " ", full_text)
    text = re.sub(r"\n{2,}", "\n", text)

    starters = [(m.start(), m.group(1)) for m in RE_ANY_STARTER.finditer(text)]
    if not starters:
        return []

    chunks: List[Tuple[str, str]] = []
    for i, (pos, w) in enumerate(starters):
        # Only accept relation starters for edge extraction
        if not RE_REL_STARTER.match(w):
            continue
        rel = rel_base(w)
        if not rel:
            continue

        end = starters[i + 1][0] if i + 1 < len(starters) else len(text)
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append((chunk, rel))
    return chunks

# =========================
# Citation symbol extraction
# =========================
def normalize_symbol(sym: str) -> str:
    sym = sym.strip().upper()
    sym = re.sub(r"\s+", "", sym)
    return sym

def cited_symbols(text: str) -> List[str]:
    out: Set[str] = set()

    # 1) Direct A/RES/.. pattern
    for m in RE_A_RES.finditer(text):
        out.add(normalize_symbol(m.group(0)))

    # 2) "resolution 2625 (XXV)" -> A/RES/2625(XXV)
    for m in RE_RES_NUM_ROMAN.finditer(text):
        num = m.group(1)
        roman = m.group(2).upper()
        out.add(f"A/RES/{num}({roman})")

    # 3) "General Assembly resolution 70/1" / "resolution 70/1"
    for m in RE_GA_RES_SLASH.finditer(text):
        out.add(f"A/RES/{m.group(1)}/{m.group(2)}")
    for m in RE_RES_SLASH.finditer(text):
        out.add(f"A/RES/{m.group(1)}/{m.group(2)}")

    # 4) Enumerations after "resolutions" (broad) — but restrict via context "of"
    #    to avoid picking "3/23" style page counter fragments that survived.
    if RE_RES_LIST_HEAD.search(text):
        for m in RE_RES_LIST_HEAD.finditer(text):
            tail = text[m.end():]
            tail = tail[:900]

            # Prefer tokens that are followed by "of" (common in cited lists)
            for t in RE_SESS_SLASH_OF.finditer(tail):
                sess = t.group(1)
                num = t.group(2)
                out.add(f"A/RES/{sess}/{num}")

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
        raise FileNotFoundError(f"No PDFs found in: {PDF_DIR}")

    # For edge de-duplication
    seen_edges: Set[Tuple[str, str, str]] = set()
    edges_written = 0

    # Clause distribution counters
    clause_counter = Counter()               # clause_starter -> num clauses
    clause_with_cite = Counter()             # clause_starter -> num clauses that contain any cited symbols
    clause_total_cite_edges = Counter()      # clause_starter -> total cited symbols occurrences (counted as edges)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["citing_symbol", "cited_symbol", "relation", "evidence_snippet", "pdf_file"])

        for i, pdf in enumerate(files, 1):
            citing_symbol = pdf.stem.replace("_", "/").upper()

            try:
                raw = extract_text_pypdf(pdf)
            except Exception:
                continue

            # Clean + join
            lines = clean_lines(raw)
            text = "\n".join(lines)

            # ---- A) Clause distribution on ALL clause starters ----
            clauses = extract_all_clauses(text)
            for chunk_text, starter in clauses:
                clause_counter[starter] += 1
                cited = cited_symbols(chunk_text)
                if cited:
                    clause_with_cite[starter] += 1
                    clause_total_cite_edges[starter] += len(cited)

            # ---- B) Edge extraction only from recalling/reaffirming/reiterating clauses ----
            rel_chunks = extract_relation_chunks(text)
            for chunk_text, rel in rel_chunks:
                cited = cited_symbols(chunk_text)
                if not cited:
                    continue

                for c in cited:
                    if c == citing_symbol:
                        continue
                    key = (citing_symbol, c, rel)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    w.writerow([citing_symbol, c, rel, snippet(chunk_text), pdf.name])
                    edges_written += 1

            if i % 500 == 0:
                print("processed pdf", i, "edges", edges_written)

    # ---- write clause distribution summary ----
    with OUT_REL_SUMMARY_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "clause_starter",
            "num_clauses",
            "clauses_with_citations",
            "total_cited_edges",
            "citation_ratio",
        ])
        for starter in sorted(clause_counter.keys()):
            total = clause_counter[starter]
            cited_n = clause_with_cite.get(starter, 0)
            edges_n = clause_total_cite_edges.get(starter, 0)
            ratio = (cited_n / total) if total else 0.0
            w.writerow([starter, total, cited_n, edges_n, f"{ratio:.4f}"])

    rel_json = {
        "pdf_count": len(files),
        "edges_written": edges_written,
        "edges_output": str(OUT_CSV),
        "method_edges": "relation_clause_chunks (recall/reaffirm/reiterate only; unspecified not produced)",
        "clause_distribution_output_tsv": str(OUT_REL_SUMMARY_TSV),
        "clause_distribution": {
            "num_clause_types": len(clause_counter),
            "top10_by_cited_edges": sorted(
                [{"starter": k, "total_cited_edges": int(v), "clauses": int(clause_counter[k]), "clauses_with_citations": int(clause_with_cite.get(k, 0))}
                 for k, v in clause_total_cite_edges.items()],
                key=lambda x: (-x["total_cited_edges"], x["starter"])
            )[:10],
        },
    }
    OUT_REL_SUMMARY_JSON.write_text(json.dumps(rel_json, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_SUMMARY.write_text(json.dumps({
        "pdf_count": len(files),
        "edges_written": edges_written,
        "output": str(OUT_CSV),
        "method": "relation_clause_chunks",
        "note": "edges are extracted only from recalling/reaffirming/reiterating chunks; unspecified is intentionally not produced here",
        "clause_distribution_tsv": str(OUT_REL_SUMMARY_TSV),
        "clause_distribution_json": str(OUT_REL_SUMMARY_JSON),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print("edges", edges_written)
    print("edges output:", OUT_CSV)
    print("clause distribution:", OUT_REL_SUMMARY_TSV)

if __name__ == "__main__":
    main()