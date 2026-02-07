#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
EDGES_CSV = DUMP_DIR / "edges.csv"
TITLES_TSV = DUMP_DIR / "symbol_title.tsv"
OUT_DIR = DUMP_DIR / "lineplot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= relation classification =========
EXCLUDE_RELATIONS = {"unspecified"}

VALUE_NEUTRAL_KEYS = {
    "recall", "recalls", "recalling",
    "note", "noting", "takingnote",
    "recognize", "recognizing", "recognized",
    "bearinginmind", "mindful",
    "havingconsidered",
}

STANCE_KEYS = {
    "reaffirm", "reaffirms", "reaffirming", "reaffirmed",
    "reiterate", "reiterates", "reiterating", "reiterated",
    "affirm", "affirms", "affirming", "affirmed",
    "welcome", "welcomes", "welcoming", "welcomed",
    "stress", "stresses", "stressing", "stressed",
    "emphasize", "emphasizes", "emphasizing", "emphasized",
    "concerned",
    "acknowledge", "acknowledges", "acknowledging", "acknowledged",
}

# ========= helpers =========
def norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "")

def norm_sym(s: str) -> str:
    return (s or "").strip().upper()

def get_session(sym: str) -> int | None:
    try:
        return int(sym.split("/")[2])
    except Exception:
        return None

def rel_bucket(rel: str) -> str | None:
    r = norm(rel)
    if not r or r in EXCLUDE_RELATIONS:
        return None
    if r in VALUE_NEUTRAL_KEYS:
        return "value_neutral"
    if r in STANCE_KEYS:
        return "stance"
    return "other"

def load_titles() -> Dict[str, str]:
    titles = {}
    if not TITLES_TSV.exists():
        return titles
    with TITLES_TSV.open(encoding="utf-8") as f:
        _ = f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2:
                titles[norm_sym(p[0])] = p[1].strip()
    return titles

# ========= load edges =========
def load_edges() -> List[Tuple[str, str, str]]:
    """
    return: [(citing, cited, bucket)]
    """
    out = []
    with EDGES_CSV.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            citing = norm_sym(row["citing_symbol"])
            cited = norm_sym(row["cited_symbol"])
            rel = row.get("relation", "")
            if not citing or not cited:
                continue
            bucket = rel_bucket(rel)
            if bucket is None:
                continue
            out.append((citing, cited, bucket))
    return out

# ========= core =========
def build_indeg_time_series(
    edges: List[Tuple[str, str, str]],
    sessions: List[int],
    mode: str,
) -> Dict[str, Dict[int, int]]:
    """
    mode: all | value_neutral | stance
    return:
      symbol -> {session -> indeg}
    """
    series = defaultdict(lambda: defaultdict(int))

    for citing, cited, bucket in edges:
        s = get_session(citing)
        if s not in sessions:
            continue

        if mode != "all" and bucket != mode:
            continue

        series[cited][s] += 1

    return series

def pick_top10(series: Dict[str, Dict[int, int]]) -> List[str]:
    totals = {
        sym: sum(v.values())
        for sym, v in series.items()
    }
    return [
        sym for sym, _ in
        sorted(totals.items(), key=lambda x: -x[1])[:10]
    ]

def extract_yearly_hot_topics(
    series: Dict[str, Dict[int, int]],
    sessions: List[int],
    global_top: set[str],
    top_k: int = 10,
) -> List[Tuple[int, str, int]]:
    rows = []

    for s in sessions:
        yearly = [
            (sym, m.get(s, 0))
            for sym, m in series.items()
            if m.get(s, 0) > 0
        ]
        yearly_sorted = sorted(yearly, key=lambda x: -x[1])[:top_k]

        for sym, cnt in yearly_sorted:
            if sym not in global_top:
                rows.append((s, sym, cnt))

    return rows

# ========= plotting =========
def plot_lines(
    series: Dict[str, Dict[int, int]],
    symbols: List[str],
    sessions: List[int],
    titles: Dict[str, str],
    out_png: pathlib.Path,
    title_suffix: str,
):
    plt.figure(figsize=(14, 8))

    for sym in symbols:
        y = [series.get(sym, {}).get(s, 0) for s in sessions]
        label = f"{sym} — {titles.get(sym, '')}"
        plt.plot(sessions, y, marker="o", linewidth=2, label=label)

    plt.xlabel("Session")
    plt.ylabel("In-degree (citations)")
    plt.title(f"Top 10 Most-Cited Resolutions ({title_suffix})")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ========= main =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", type=int, required=True, help="latest session (e.g. 79)")
    ap.add_argument("--years", type=int, default=10, help="how many years to look back")
    args = ap.parse_args()

    sessions = list(range(args.X - args.years + 1, args.X + 1))

    titles = load_titles()
    edges = load_edges()

    for mode, label in [
        ("all", "ALL"),
        ("value_neutral", "VALUE_NEUTRAL"),
        ("stance", "STANCE"),
    ]:
        series = build_indeg_time_series(edges, sessions, mode)
        global_top10 = set(pick_top10(series))

        hot_rows = extract_yearly_hot_topics(
            series=series,
            sessions=sessions,
            global_top=global_top10,
            top_k=10,
        )

        out_csv = OUT_DIR / f"hot_topics_{label.lower()}_last_{args.years}_years.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["session", "cited_symbol", "in_degree", "title"])
            for s, sym, cnt in hot_rows:
                w.writerow([s, sym, cnt, titles.get(sym, "")])

        print(f"[{label}] hot-topic CSV saved:", out_csv)

        top10 = pick_top10(series)
        out_png = OUT_DIR / f"top10_{label.lower()}_last_{args.years}_years.png"

        plot_lines(
            series=series,
            symbols=top10,
            sessions=sessions,
            titles=titles,
            out_png=out_png,
            title_suffix=f"{label} citations, last {args.years} years",
        )

        print(f"[{label}] saved:", out_png)

    print("done")

if __name__ == "__main__":
    main()