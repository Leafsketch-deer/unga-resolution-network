#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
EDGES_CSV = DUMP_DIR / "edges_reclassified.csv"
TITLES_TSV = DUMP_DIR / "symbol_title.tsv"
OUT_DIR = DUMP_DIR / "lineplot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= config =========
EXCLUDE_RELATIONS = {"unspecified"}  # 念のため（reclassified後は出ない想定でもガード）

MODES = [
    ("all", "ALL"),
    ("value_neutral", "VALUE_NEUTRAL"),
    ("stance", "STANCE"),
]


# ========= helpers =========
def norm_sym(s: str) -> str:
    return (s or "").strip().upper()


def get_session(sym: str) -> Optional[int]:
    try:
        # A/RES/72/17 -> 72
        return int(sym.split("/")[2])
    except Exception:
        return None


def load_titles() -> Dict[str, str]:
    titles: Dict[str, str] = {}
    if not TITLES_TSV.exists():
        return titles

    with TITLES_TSV.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # header想定
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2:
                titles[norm_sym(p[0])] = p[1].strip()
    return titles


def load_edges() -> List[Tuple[str, str, str, str]]:
    """
    edges_reclassified.csv を読む
    return: [(citing_symbol, cited_symbol, relation, relation_group)]
    """
    if not EDGES_CSV.exists():
        raise FileNotFoundError(EDGES_CSV)

    out: List[Tuple[str, str, str, str]] = []
    with EDGES_CSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            citing = norm_sym(row.get("citing_symbol", ""))
            cited = norm_sym(row.get("cited_symbol", ""))
            rel = (row.get("relation") or "").strip()
            grp = (row.get("relation_group") or "").strip()

            if not citing or not cited:
                continue
            if not grp:
                continue
            if (rel or "").strip().lower().replace(" ", "") in EXCLUDE_RELATIONS:
                continue

            # grp は "value_neutral" / "stance" / "other" を想定
            out.append((citing, cited, rel, grp))
    return out


# ========= core =========
def build_indeg_time_series(
    edges: List[Tuple[str, str, str, str]],
    sessions: List[int],
    mode: str,
) -> Dict[str, Dict[int, int]]:
    """
    mode: all | value_neutral | stance
    return: cited_symbol -> {session -> indeg}
    """
    series: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for citing, cited, _rel, grp in edges:
        s = get_session(citing)
        if s is None or s not in sessions:
            continue

        if mode != "all" and grp != mode:
            continue

        series[cited][s] += 1

    return series


def pick_topk(series: Dict[str, Dict[int, int]], k: int = 10) -> List[str]:
    totals = {sym: sum(m.values()) for sym, m in series.items()}
    return [sym for sym, _ in sorted(totals.items(), key=lambda x: (-x[1], x[0]))[:k]]


def extract_yearly_hot_topics(
    series: Dict[str, Dict[int, int]],
    sessions: List[int],
    global_top: set[str],
    top_k: int = 10,
) -> List[Tuple[int, str, int]]:
    """
    期間合計Topには入らないが、各sessionのTopKに入るものを抽出
    return: [(session, symbol, indeg), ...]
    """
    rows: List[Tuple[int, str, int]] = []
    for s in sessions:
        yearly = [(sym, m.get(s, 0)) for sym, m in series.items() if m.get(s, 0) > 0]
        yearly_sorted = sorted(yearly, key=lambda x: (-x[1], x[0]))[:top_k]
        for sym, cnt in yearly_sorted:
            if sym not in global_top:
                rows.append((s, sym, cnt))
    return rows


# ========= color policy =========
def build_color_policy(
    top10_all: List[str],
    extra_symbols: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    要件：
    - all の top10 は固定色（濃いめ）
    - それ以外は top10 と被らない色から選ぶ
    - それ以外は「薄い」色にする
    """
    # 1) top10用：tab10（最大10色で安定）
    base_cmap = cm.get_cmap("tab10")
    top10_color_map: Dict[str, str] = {
        sym: mcolors.to_hex(base_cmap(i))
        for i, sym in enumerate(top10_all)
    }
    used_hex = set(top10_color_map.values())

    # 2) それ以外：tab20 から「未使用色」だけ拾って、薄くする
    # tab20は20色。足りなければ繰り返しで回す（ただし薄色なので識別はある程度可能）
    extra_cmap = cm.get_cmap("tab20")
    extra_palette = []
    for i in range(extra_cmap.N):
        hx = mcolors.to_hex(extra_cmap(i))
        if hx not in used_hex:
            extra_palette.append(hx)

    def lighten_hex(hex_color: str, amount: float = 0.55) -> str:
        """
        amount: 0 -> 元色のまま, 1 -> 白
        """
        r, g, b = mcolors.to_rgb(hex_color)
        r = r + (1.0 - r) * amount
        g = g + (1.0 - g) * amount
        b = b + (1.0 - b) * amount
        return mcolors.to_hex((r, g, b))

    extra_color_map: Dict[str, str] = {}
    if extra_palette:
        for idx, sym in enumerate(extra_symbols):
            base = extra_palette[idx % len(extra_palette)]
            extra_color_map[sym] = lighten_hex(base, amount=0.55)
    else:
        # 万一すべて被った場合（通常起きない）: グレー薄め
        for sym in extra_symbols:
            extra_color_map[sym] = lighten_hex("#808080", amount=0.6)

    return top10_color_map, extra_color_map


# ========= plotting =========
def plot_lines(
    series: Dict[str, Dict[int, int]],
    symbols: List[str],
    sessions: List[int],
    titles: Dict[str, str],
    out_png: pathlib.Path,
    title_suffix: str,
    top10_color_map: Dict[str, str],
    extra_color_map: Dict[str, str],
    legend_loc: str = "upper left",
) -> None:
    plt.figure(figsize=(14, 8))

    for sym in symbols:
        y = [series.get(sym, {}).get(s, 0) for s in sessions]
        label = f"{sym} — {titles.get(sym, '')}".strip()

        # 色の決定（top10優先）
        color = top10_color_map.get(sym)
        if color is None:
            color = extra_color_map.get(sym)

        # top10以外は薄色なので、線も少し細く＆透明度を少し下げる
        is_top10 = sym in top10_color_map
        lw = 2.2 if is_top10 else 1.8
        alpha = 1.0 if is_top10 else 0.75

        plt.plot(
            sessions,
            y,
            marker="o",
            linewidth=lw,
            alpha=alpha,
            label=label,
            color=color,  # Noneならmatplotlibに任せるが、ここでは基本Noneにならない設計
        )

    plt.xlabel("Session")
    plt.ylabel("In-degree (citations)")
    plt.title(f"Top 10 Most-Cited Resolutions ({title_suffix})")

    # 凡例（左上固定）
    plt.legend(fontsize=8, loc=legend_loc)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ========= main =========
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", type=int, required=True, help="latest session (e.g. 79)")
    ap.add_argument("--years", type=int, default=10, help="how many years to look back")
    ap.add_argument("--legend-loc", type=str, default="upper left", help="matplotlib legend loc")
    args = ap.parse_args()

    sessions = list(range(args.X - args.years + 1, args.X + 1))

    titles = load_titles()
    edges = load_edges()

    # --- 色固定の基準：allのtop10 ---
    series_all = build_indeg_time_series(edges, sessions, "all")
    top10_all = pick_topk(series_all, k=10)

    # 以後のグラフに出る可能性がある「追加シンボル候補」を集める
    # （value_neutral / stance の top10 が all と完全一致しない場合に備える）
    extra_symbols_set = set()

    # 先に各モードのtop10を確定しておく
    mode_to_top10: Dict[str, List[str]] = {}
    mode_to_series: Dict[str, Dict[str, Dict[int, int]]] = {}

    for mode, _label in MODES:
        series = build_indeg_time_series(edges, sessions, mode)
        mode_to_series[mode] = series
        top10 = pick_topk(series, k=10)
        mode_to_top10[mode] = top10
        for sym in top10:
            if sym not in top10_all:
                extra_symbols_set.add(sym)

    extra_symbols = sorted(extra_symbols_set)
    top10_color_map, extra_color_map = build_color_policy(top10_all=top10_all, extra_symbols=extra_symbols)

    # --- 各モードの出力 ---
    for mode, label in MODES:
        series = mode_to_series[mode]
        top10 = mode_to_top10[mode]
        global_top10 = set(top10)

        # hot topics CSV
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

        # lineplot
        out_png = OUT_DIR / f"top10_{label.lower()}_last_{args.years}_years.png"
        plot_lines(
            series=series,
            symbols=top10,
            sessions=sessions,
            titles=titles,
            out_png=out_png,
            title_suffix=f"{label} citations, last {args.years} years",
            top10_color_map=top10_color_map,
            extra_color_map=extra_color_map,
            legend_loc=args.legend_loc,
        )
        print(f"[{label}] saved:", out_png)

    print("done")
    print("color policy:")
    print("  fixed(top10_all):", len(top10_color_map))
    print("  extra(thin):", len(extra_color_map))


if __name__ == "__main__":
    main()