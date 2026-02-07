#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
IN_EDGES = DUMP_DIR / "edges.csv"

OUT_RECLASS = DUMP_DIR / "edges_reclassified.csv"
OUT_BREAKDOWN = DUMP_DIR / "relation_breakdown.tsv"

# ========= classification =========

VALUE_NEUTRAL = {
    "recall", "recalls", "recalled", "recalling",
    "noting", "taking note", "bearing in mind", "mindful",
    "having considered", "guided by",
}

STANCE = {
    "reaffirm", "reaffirms", "reaffirmed", "reaffirming",
    "affirm", "affirms", "affirmed", "affirming",
    "reiterate", "reiterates", "reiterated", "reiterating",
    "stress", "stresses", "stressed", "stressing",
    "emphasize", "emphasizes", "emphasized", "emphasizing",
    "welcome", "welcomes", "welcomed", "welcoming",
    "recognize", "recognizes", "recognized", "recognizing",
    "acknowledge", "acknowledges", "acknowledged", "acknowledging",
    "concerned",
}

def normalize_rel(r: str) -> str:
    return (r or "").strip().lower()

def classify_relation(r: str) -> str:
    r = normalize_rel(r)
    if r in VALUE_NEUTRAL:
        return "value_neutral"
    if r in STANCE:
        return "stance"
    return "other"

# ========= main =========
def main():
    if not IN_EDGES.exists():
        raise FileNotFoundError(IN_EDGES)

    breakdown = defaultdict(lambda: {
        "edges": 0,
        "value_neutral": 0,
        "stance": 0,
        "other": 0,
    })

    with IN_EDGES.open("r", encoding="utf-8", errors="replace", newline="") as f_in, \
         OUT_RECLASS.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ["relation_group"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rel_raw = row.get("relation", "")
            rel_norm = normalize_rel(rel_raw)
            group = classify_relation(rel_norm)

            row["relation_group"] = group
            writer.writerow(row)

            breakdown[rel_norm]["edges"] += 1
            breakdown[rel_norm][group] += 1

    # ---- write breakdown ----
    with OUT_BREAKDOWN.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "relation_raw",
            "total_edges",
            "value_neutral",
            "stance",
            "other",
        ])
        for rel, d in sorted(breakdown.items(), key=lambda x: -x[1]["edges"]):
            w.writerow([
                rel,
                d["edges"],
                d["value_neutral"],
                d["stance"],
                d["other"],
            ])

    print("done")
    print("written:", OUT_RECLASS)
    print("written:", OUT_BREAKDOWN)

if __name__ == "__main__":
    main()