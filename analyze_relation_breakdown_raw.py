#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import re
from collections import Counter
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt

# treemap helper
try:
    import squarify  # type: ignore
    HAS_SQUARIFY = True
except Exception:
    HAS_SQUARIFY = False

# ========= defaults =========
DUMP_DIR = pathlib.Path("undl_dump")
DEFAULT_EDGES = DUMP_DIR / "edges.csv"
OUT_DIR = DUMP_DIR / "relation_breakdown"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_RELATIONS = {"unspecified"}

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

# ========= normalization =========
WS_RE = re.compile(r"\s+")

def norm_rel(rel: str) -> str:
    s = (rel or "").strip().lower()
    s = WS_RE.sub(" ", s)
    return s

def build_variant_to_lemma() -> Dict[str, str]:
    m: Dict[str, str] = {}

    # value-neutral lemmas
    m.update({v: "recall" for v in ["recall", "recalls", "recalled", "recalling"]})
    m.update({v: "noting" for v in ["noting"]})
    m["taking note"] = "taking note"
    m["bearing in mind"] = "bearing in mind"
    m["mindful"] = "mindful"
    m["having considered"] = "having considered"
    m["guided by"] = "guided by"

    # stance lemmas
    m.update({v: "reaffirm" for v in ["reaffirm", "reaffirms", "reaffirmed", "reaffirming"]})
    m.update({v: "affirm" for v in ["affirm", "affirms", "affirmed", "affirming"]})
    m.update({v: "reiterate" for v in ["reiterate", "reiterates", "reiterated", "reiterating"]})
    m.update({v: "stress" for v in ["stress", "stresses", "stressed", "stressing"]})
    m.update({v: "emphasize" for v in ["emphasize", "emphasizes", "emphasized", "emphasizing"]})
    m.update({v: "welcome" for v in ["welcome", "welcomes", "welcomed", "welcoming"]})
    m.update({v: "recognize" for v in ["recognize", "recognizes", "recognized", "recognizing"]})
    m.update({v: "acknowledge" for v in ["acknowledge", "acknowledges", "acknowledged", "acknowledging"]})
    m["concerned"] = "concerned"

    return m

VARIANT_TO_LEMMA = build_variant_to_lemma()

def lemma_of(rel: str) -> str:
    r = norm_rel(rel)
    return VARIANT_TO_LEMMA.get(r, r)

def iter_relations(edges_csv: pathlib.Path) -> Iterable[str]:
    with edges_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rel = norm_rel(row.get("relation", "") or "")
            if not rel or rel in EXCLUDE_RELATIONS:
                continue
            yield rel

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ========= output helpers =========
def write_tsv(path: pathlib.Path, items: list[Tuple[str, int]]) -> None:
    total = sum(v for _, v in items) if items else 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["lemma_relation", "count", "share_percent"])
        for k, v in items:
            pct = (v / total * 100.0) if total else 0.0
            w.writerow([k, v, f"{pct:.6f}"])

# ========= treemap =========
def plot_treemap(items: list[Tuple[str, int]], out_png: pathlib.Path, title: str) -> None:
    if not HAS_SQUARIFY:
        raise RuntimeError("squarify is not installed. Please: pip install squarify")

    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    total = sum(counts) if counts else 0
    if total <= 0:
        raise RuntimeError("No data to plot (counts sum to 0).")

    # ラベルに比率も入れる（見やすさ優先）
    disp_labels = []
    for k, v in items:
        pct = v / total * 100.0
        disp_labels.append(f"{k}\n{pct:.1f}% (n={v})")

    plt.figure(figsize=(12, 7))
    squarify.plot(sizes=counts, label=disp_labels, alpha=0.9)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=pathlib.Path, default=DEFAULT_EDGES, help="path to edges.csv")
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR, help="output directory")
    ap.add_argument("--top", type=int, default=30, help="top N relations to show (after lemmatization)")
    ap.add_argument("--include-other", action="store_true",
                    help="include relations not in VALUE_NEUTRAL/STANCE as 'other' group")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    raw_counter = Counter()
    lemma_counter = Counter()

    for rel in iter_relations(args.edges):
        raw_counter[rel] += 1
        lemma_counter[lemma_of(rel)] += 1

    known_lemmas = set(lemma_of(x) for x in (VALUE_NEUTRAL | STANCE))

    if args.include_other:
        other_sum = 0
        collapsed = Counter()
        for k, v in lemma_counter.items():
            if k in known_lemmas:
                collapsed[k] += v
            else:
                other_sum += v
        if other_sum > 0:
            collapsed["other"] = other_sum
        lemma_counter = collapsed
    else:
        lemma_counter = Counter({k: v for k, v in lemma_counter.items() if k in known_lemmas})

    items = sorted(lemma_counter.items(), key=lambda x: (-x[1], x[0]))[:max(args.top, 1)]

    out_tsv = args.out_dir / "relation_lemma_ratio.tsv"
    out_png = args.out_dir / "relation_lemma_ratio_treemap.png"

    write_tsv(out_tsv, items)

    title = "Relation breakdown (treemap; lemmatized; unspecified excluded)"
    if args.include_other:
        title += " + other"
    plot_treemap(items, out_png, title=title)

    print("done")
    print("wrote:", out_tsv)
    print("plot :", out_png)

    if not args.include_other:
        total_all = sum(raw_counter.values())
        total_known = sum(lemma_counter.values())
        if total_all > 0:
            print(f"known share (among all non-unspecified edges): {total_known}/{total_all} = {total_known/total_all:.3%}")

if __name__ == "__main__":
    main()