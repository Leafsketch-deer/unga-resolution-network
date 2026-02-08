#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
EDGES_CSV = DUMP_DIR / "edges_reclassified.csv"
OUT_DIR = DUMP_DIR / "cdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= helpers =========
def norm_sym(s: str) -> str:
    return (s or "").strip().upper()

def load_edges() -> List[dict]:
    rows = []
    with EDGES_CSV.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("relation_group"):
                continue
            rows.append(row)
    return rows

def build_indegree(
    rows: List[dict],
    mode: str,
) -> Dict[str, int]:
    """
    mode:
      - all
      - value_neutral
      - stance
    """
    indeg = defaultdict(int)

    for row in rows:
        g = row["relation_group"]
        if mode != "all" and g != mode:
            continue

        cited = norm_sym(row["cited_symbol"])
        if cited:
            indeg[cited] += 1

    return indeg

def compute_cdf(values: List[int]):
    """
    return:
      x: sorted unique indeg
      y: cumulative ratio
    """
    values = sorted(values)
    n = len(values)

    xs = []
    ys = []

    cum = 0
    for v in values:
        cum += 1
        xs.append(v)
        ys.append(cum / n)

    return xs, ys

# ========= main =========
def main():
    rows = load_edges()

    plt.figure(figsize=(10, 7))

    for mode, label, color in [
        ("all", "ALL", "black"),
        ("value_neutral", "VALUE_NEUTRAL", "blue"),
        ("stance", "STANCE", "red"),
    ]:
        indeg = build_indegree(rows, mode)
        xs, ys = compute_cdf(list(indeg.values()))

        plt.plot(
            xs,
            ys,
            label=label,
            linewidth=2,
            color=color,
        )

    plt.xscale("log")
    plt.xlabel("In-degree (log scale)")
    plt.ylabel("Cumulative fraction of resolutions")
    plt.title("CDF of In-degree (UNGA Resolution Citations)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = OUT_DIR / "indegree_cdf_all_vs_groups.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("saved:", out_png)

if __name__ == "__main__":
    main()