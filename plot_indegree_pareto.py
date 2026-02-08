#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
EDGES_RECLASSIFIED_CSV = DUMP_DIR / "edges_reclassified.csv"
TITLES_TSV = DUMP_DIR / "symbol_title.tsv"
OUT_DIR = DUMP_DIR / "pareto_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 50
TOP_LEGEND = 10


# ========= helpers =========
def norm_sym(s: str) -> str:
    return (s or "").strip().upper()

def load_titles(tsv_path: pathlib.Path) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    if not tsv_path.exists():
        return titles

    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2:
                sym = norm_sym(p[0])
                title = p[1].strip()
                if sym and title and sym not in titles:
                    titles[sym] = title
    return titles

def load_edges_counts_by_group(csv_path: pathlib.Path) -> Dict[str, Dict[str, int]]:
    counts_all = defaultdict(int)
    counts_neu = defaultdict(int)
    counts_sta = defaultdict(int)

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cited = norm_sym(row.get("cited_symbol", ""))
            g = (row.get("relation_group") or "").strip()
            if not cited or g == "unspecified":
                continue

            counts_all[cited] += 1
            if g == "value_neutral":
                counts_neu[cited] += 1
            elif g == "stance":
                counts_sta[cited] += 1

    return {
        "all": dict(counts_all),
        "value_neutral": dict(counts_neu),
        "stance": dict(counts_sta),
    }

def top_items(counts: Dict[str, int], n: int) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]

def make_top10_legend_text(
    items: List[Tuple[str, int]],
    titles: Dict[str, str],
    max_title_len: int = 60,
) -> str:
    lines = []
    for i, (sym, _) in enumerate(items[:TOP_LEGEND], 1):
        t = titles.get(sym, "")
        if t and len(t) > max_title_len:
            t = t[: max_title_len - 1] + "…"
        if t:
            lines.append(f"{i}. {sym}\n   {t}")
        else:
            lines.append(f"{i}. {sym}")
    return "\n".join(lines)

def plot_pareto(
    items: List[Tuple[str, int]],
    titles: Dict[str, str],
    out_png: pathlib.Path,
    chart_title: str,
    grand_total: int,   # ★分母（全被言及数）
) -> None:
    values = [cnt for _, cnt in items]

    # ★分母は “上位50合計” ではなく “全体合計”
    denom = grand_total if grand_total > 0 else 1

    cum_pct = []
    acc = 0
    for v in values:
        acc += v
        cum_pct.append((acc / denom) * 100)

    fig, ax1 = plt.subplots(figsize=(16, 8))
    x = list(range(len(values)))

    # ---- bar (in-degree) ----
    ax1.bar(
        x,
        values,
        color="#1f77b4",   # blue
        alpha=0.85,
        label="In-degree (Top shown)",
    )
    ax1.set_ylabel("In-degree (count)")
    ax1.set_xlabel(f"Top {len(values)} resolutions (ranked)")
    ax1.set_xticks([])

    # ---- cumulative line ----
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        cum_pct,
        color="#ff7f0e",   # orange
        marker="o",
        linewidth=2,
        label="Cumulative % (denom = all edges)",
    )
    ax2.set_ylabel("Cumulative percentage (%)")
    ax2.set_ylim(0, 100)
    ax2.axhline(80, linestyle="--", color="gray", alpha=0.5)

    ax1.set_title(chart_title)

    # ---- legend box (Top10 resolutions) ----
    legend_text = make_top10_legend_text(items, titles)
    ax1.text(
        0.99, 0.50,                 # right-center
        legend_text,
        transform=ax1.transAxes,
        fontsize=15,
        va="center",
        ha="right",
        multialignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="gray",
        ),
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    titles = load_titles(TITLES_TSV)
    counts_by_group = load_edges_counts_by_group(EDGES_RECLASSIFIED_CSV)

    for mode, label in [
        ("all", "ALL"),
        ("value_neutral", "VALUE_NEUTRAL"),
        ("stance", "STANCE"),
    ]:
        counts = counts_by_group.get(mode, {})
        items = top_items(counts, TOP_N)

        # ★全体分母（このmodeにおける全決議の被言及数合計）
        grand_total = sum(counts.values())

        out_png = OUT_DIR / f"pareto_top{TOP_N}_{mode}.png"

        plot_pareto(
            items=items,
            titles=titles,
            out_png=out_png,
            chart_title=f"Pareto chart (Top {TOP_N}) — {label}\nCumulative % denom = all citations in group",
            grand_total=grand_total,
        )
        print("saved:", out_png, "(grand_total =", grand_total, ")")

    print("done")


if __name__ == "__main__":
    main()