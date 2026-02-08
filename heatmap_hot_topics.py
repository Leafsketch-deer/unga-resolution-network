#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
LINEPLOT_DIR = DUMP_DIR / "lineplot"

INPUTS = {
    "all": LINEPLOT_DIR / "hot_topics_all_last_10_years.csv",
    "stance": LINEPLOT_DIR / "hot_topics_stance_last_10_years.csv",
    "value_neutral": LINEPLOT_DIR / "hot_topics_value_neutral_last_10_years.csv",
}

OUTS = {
    "all": LINEPLOT_DIR / "heatmap_hot_topics_all_last_10_years.png",
    "stance": LINEPLOT_DIR / "heatmap_hot_topics_stance_last_10_years.png",
    "value_neutral": LINEPLOT_DIR / "heatmap_hot_topics_value_neutral_last_10_years.png",
}


def load_hot_topics_csv(path: pathlib.Path) -> pd.DataFrame:
    """
    expected columns:
      session, cited_symbol, in_degree, title
    """
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "session": int(row["session"]),
                "symbol": (row["cited_symbol"] or "").strip(),
                "in_degree": int(row["in_degree"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df


def make_heatmap(df: pd.DataFrame, out_png: pathlib.Path, title: str) -> None:
    """
    - y: symbol (sorted by total in_degree desc)
    - x: session
    - cell: in_degree (missing -> 0)
    """
    if df.empty:
        print("skip (empty):", out_png)
        return

    pivot = (
        df.pivot_table(
            index="symbol",
            columns="session",
            values="in_degree",
            aggfunc="sum",
            fill_value=0
        )
    )

    # sort rows by total desc (to keep "bigger" ones near top)
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total__", ascending=False).drop(columns="__total__")

    # figure height depends on number of symbols
    fig_h = max(6, 0.35 * len(pivot))
    plt.figure(figsize=(14, fig_h))

    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="lightgray",
        cbar_kws={"label": "In-degree (citations)"},
    )

    plt.xlabel("Session")
    plt.ylabel("Resolution (cited_symbol)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("saved:", out_png)


def main() -> None:
    # seaborn can be missing depending on env; if so, error clearly
    try:
        _ = sns.__version__
    except Exception as e:
        raise RuntimeError("seaborn is required. Install with: pip install seaborn") from e

    for key in ["all", "stance", "value_neutral"]:
        in_csv = INPUTS[key]
        out_png = OUTS[key]

        df = load_hot_topics_csv(in_csv)
        make_heatmap(
            df=df,
            out_png=out_png,
            title=f"Yearly Hot Topics ({key}) — Heatmap of in-degree by session (last 10 years)",
        )


if __name__ == "__main__":
    main()