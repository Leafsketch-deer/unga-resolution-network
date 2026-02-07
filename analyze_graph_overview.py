#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")

# 新
EDGES_CSV = DUMP_DIR / "edges_reclassified.csv"
TITLES_TSV = DUMP_DIR / "symbol_title.tsv"   # symbol \t title \t ...
OUT_DIR = DUMP_DIR / "overview_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= config =========
# 0. 前提：unspecifiedは除外
EXCLUDE_RELATIONS = {"unspecified"}


# ========= helpers =========
def norm_sym(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().upper())

def norm_rel(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())

def rel_bucket(row: dict) -> Optional[str]:
    """
    returns: "value_neutral" / "stance" / "other" / None
    """
    g = (row.get("relation_group") or "").strip()
    if not g:
        return None
    if g == "other":
        return "other"
    return g

def load_titles(tsv_path: pathlib.Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not tsv_path.exists():
        return mp

    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # header想定（無くても害はない）
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sym = norm_sym(parts[0])
            title = parts[1].strip()
            if sym and title:
                mp.setdefault(sym, title)
    return mp

@dataclass
class Edge:
    citing: str
    cited: str
    rel: str
    bucket: str   # weak/strong/other

def iter_edges(edges_csv: pathlib.Path) -> Iterable[Edge]:
    with edges_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            citing = norm_sym(row.get("citing_symbol", ""))
            cited  = norm_sym(row.get("cited_symbol", ""))
            rel    = row.get("relation", "") or ""

            if not citing or not cited:
                continue

            b = rel_bucket(row)
            if b is None:
                continue

            yield Edge(
                citing=citing,
                cited=cited,
                rel=norm_rel(rel),
                bucket=b,   # value_neutral / stance / other
            )

def safe_mean(xs: List[int]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0

def safe_median(xs: List[int]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return (ys[mid - 1] + ys[mid]) / 2.0

def degree_stats(values: Dict[str, int]) -> Dict[str, float]:
    xs = list(values.values())
    if not xs:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": int(min(xs)),
        "max": int(max(xs)),
        "mean": float(safe_mean(xs)),
        "median": float(safe_median(xs)),
    }

def write_json(path: pathlib.Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_top10_tsv(path: pathlib.Path, rows: List[Tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "symbol", "title", "in_degree"])
        for i, (sym, title, deg) in enumerate(rows, 1):
            w.writerow([i, sym, title, deg])

def plot_in_degree(sorted_items: List[Tuple[str, int]], out_png: pathlib.Path, title: str) -> None:
    ys = [deg for _, deg in sorted_items]
    xs = list(range(1, len(ys) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys)
    plt.yscale("log")
    plt.xlabel("Resolutions (sorted by in-degree desc)")
    plt.ylabel("In-degree (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main() -> None:
    if not EDGES_CSV.exists():
        raise FileNotFoundError(EDGES_CSV)

    titles = load_titles(TITLES_TSV)

    out_all = defaultdict(int)
    in_all  = defaultdict(int)

    out_neutral = defaultdict(int)   # value_neutral
    in_neutral  = defaultdict(int)

    out_stance = defaultdict(int)    # stance
    in_stance  = defaultdict(int)

    total_edges = 0
    total_edges_neutral = 0
    total_edges_stance = 0
    total_edges_other = 0

    nodes = set()

    for e in iter_edges(EDGES_CSV):
        total_edges += 1
        nodes.add(e.citing)
        nodes.add(e.cited)

        # a: unspecified除外済みの総数（= value_neutral / stance / other 全て）
        out_all[e.citing] += 1
        in_all[e.cited] += 1

        # b/c: 新カテゴリで集計
        if e.bucket == "value_neutral":
            total_edges_neutral += 1
            out_neutral[e.citing] += 1
            in_neutral[e.cited] += 1
        elif e.bucket == "stance":
            total_edges_stance += 1
            out_stance[e.citing] += 1
            in_stance[e.cited] += 1
        else:
            total_edges_other += 1

    overview = {
        "inputs": {
            "edges_csv": str(EDGES_CSV),
            "titles_tsv": str(TITLES_TSV) if TITLES_TSV.exists() else None,
            "excluded_relations": sorted(EXCLUDE_RELATIONS),
            "relation_groups": ["value_neutral", "stance", "other"],
        },
        "counts": {
            "nodes_in_graph": len(nodes),
            "edges_total_excl_unspecified": total_edges,
            "edges_value_neutral": total_edges_neutral,
            "edges_stance": total_edges_stance,
            "edges_other": total_edges_other,
        },
        "stats": {
            "a_all_excl_unspecified": {
                "out_degree": degree_stats(out_all),
                "in_degree": degree_stats(in_all),
            },
           "b_value_neutral": {
               "out_degree": degree_stats(out_neutral),
                   "in_degree": degree_stats(in_neutral),
            },
            "c_stance": {
                "out_degree": degree_stats(out_stance),
                "in_degree": degree_stats(in_stance),
            },
        },
        "note": "Stats are computed over nodes that appear in edges.csv after excluding [unspecified]. "
                "If you want to include all resolutions (including degree=0), feed symbols.txt as the population and zero-fill.",
    }

    write_json(OUT_DIR / "overview_summary.json", overview)

    # (2) in-degree plots
    def sorted_degree_items(d: Dict[str, int]) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda x: (-x[1], x[0]))

    in_all_sorted = sorted_degree_items(in_all)
    in_neutral_sorted = sorted_degree_items(in_neutral)
    in_stance_sorted = sorted_degree_items(in_stance)

    plot_in_degree(
        in_all_sorted,
        OUT_DIR / "in_degree_all.png",
        "In-degree (all relations, excluding [unspecified])"
    )
    plot_in_degree(
        in_neutral_sorted,
        OUT_DIR / "in_degree_value_neutral.png",
        "In-degree (value-neutral references, excluding [unspecified])"
    )
    plot_in_degree(
        in_stance_sorted,
        OUT_DIR / "in_degree_stance.png",
        "In-degree (stance-expressing references, excluding [unspecified])"
    )

    # (3) top10
    def top10(d: Dict[str, int]) -> List[Tuple[str, str, int]]:
        items = sorted_degree_items(d)[:10]
        out = []
        for sym, deg in items:
            out.append((sym, titles.get(sym, ""), deg))
        return out

    write_top10_tsv(OUT_DIR / "top10_in_degree_all.tsv", top10(in_all))
    write_top10_tsv(OUT_DIR / "top10_in_degree_value_neutral.tsv", top10(in_neutral))
    write_top10_tsv(OUT_DIR / "top10_in_degree_stance.tsv", top10(in_stance))

    print("done")
    print("wrote:", OUT_DIR / "overview_summary.json")
    print("plots:", OUT_DIR / "in_degree_all.png",
          OUT_DIR / "in_degree_value_neutral.png",
          OUT_DIR / "in_degree_stance.png")
    print("top10:", OUT_DIR / "top10_in_degree_all.tsv",
          OUT_DIR / "top10_in_degree_value_neutral.tsv",
          OUT_DIR / "top10_in_degree_stance.tsv")

if __name__ == "__main__":
    main()