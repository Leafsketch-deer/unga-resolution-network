#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---- optional deps (for graph html) ----
try:
    from pyvis.network import Network  # type: ignore
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

# ---- deps for static png ----
try:
    import networkx as nx  # type: ignore
    import matplotlib.pyplot as plt
    HAS_NX = True
except Exception:
    HAS_NX = False

# ========= paths =========
DUMP_DIR = pathlib.Path("undl_dump")
EDGES_CSV = DUMP_DIR / "edges_reclassified.csv"   # ←ここを変更
TITLES_TSV = DUMP_DIR / "symbol_title.tsv"
OUT_DIR = DUMP_DIR / "trend_out"

# ========= relation buckets (must match extract_edges_from_pdfs_pypdf.py output) =========
EXCLUDE_RELATIONS = {"unspecified"}
VALID_BUCKETS = {"value_neutral", "stance", "other"}

# ========= symbol parsing =========
RE_SYM_SESSION = re.compile(r"^A/RES/(\d+)/(.*)$", re.I)

def norm_sym(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().upper())

def norm_rel(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())

def get_session(sym: str) -> Optional[int]:
    m = RE_SYM_SESSION.match(norm_sym(sym))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def rel_bucket(rel: str) -> Optional[str]:
    r = norm_rel(rel)
    if not r or r in EXCLUDE_RELATIONS:
        return None
    if r in WEAK_KEYS:
        return "weak"
    if r in STRONG_KEYS:
        return "strong"
    return "other"

def load_titles(tsv_path: pathlib.Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not tsv_path.exists():
        return mp
    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # header想定（無くてもOK）
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sym = norm_sym(parts[0])
            title = parts[1].strip()
            if sym and title:
                mp.setdefault(sym, title)
    return mp

@dataclass(frozen=True)
class Edge:
    citing: str
    cited: str
    relation: str        # 生の種類（recalling, stressed など）
    bucket: str          # value_neutral / stance / other

def iter_edges(path: pathlib.Path) -> Iterable[Edge]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            citing = norm_sym(row.get("citing_symbol", ""))
            cited  = norm_sym(row.get("cited_symbol", ""))
            rel    = norm_rel(row.get("relation", "") or "")
            if not citing or not cited:
                continue

            if rel in EXCLUDE_RELATIONS:
                continue  # unspecified除外（前提）

            b = (row.get("bucket") or row.get("relation_group") or "").strip().lower()
            if b not in VALID_BUCKETS:
                # 想定外の値は other 扱いに倒す（落とさない）
                b = "other"

            yield Edge(citing=citing, cited=cited, relation=rel, bucket=b)

# ========= core =========
# ========= core =========
def session_edges(edges: List[Edge], sess: int) -> List[Edge]:
    return [e for e in edges if get_session(e.citing) == sess]


def top10_in_degree(edges_s: List[Edge], titles: Dict[str, str], mode: str) -> List[Tuple[str, int, str]]:
    """
    mode:
      - "all": value_neutral/stance/other 全部（unspecified除外済み）
      - "value_neutral": value_neutral のみ
      - "stance": stance のみ
    """
    indeg = defaultdict(int)
    for e in edges_s:
        if mode != "all" and e.bucket != mode:
            continue
        indeg[e.cited] += 1

    items = sorted(indeg.items(), key=lambda x: (-x[1], x[0]))[:10]
    return [(sym, cnt, titles.get(sym, "")) for sym, cnt in items]


def write_top10_tsv(path: pathlib.Path, rows: List[Tuple[str, int, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "cited_symbol", "in_degree_from_session", "title"])
        for i, (sym, cnt, title) in enumerate(rows, 1):
            w.writerow([i, sym, cnt, title])


def build_subgraph(edges_s: List[Edge]) -> Tuple[Set[str], List[Edge], Dict[str, int], Dict[str, int]]:
    """
    ノード集合：会期Sのcitingノード + C(S)（citedノード）
    返り値：
      nodes,
      edges (session edges),
      indeg_from_S (cited count within S)  ※ relation問わず（unspecified除外済）
      total_degree(in+out) within S-subgraph
    """
    nodes: Set[str] = set()
    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    for e in edges_s:
        nodes.add(e.citing)
        nodes.add(e.cited)
        indeg[e.cited] += 1
        outdeg[e.citing] += 1

    total_deg = defaultdict(int)
    for n in nodes:
        total_deg[n] = indeg.get(n, 0) + outdeg.get(n, 0)

    return nodes, edges_s, indeg, total_deg


def split_indeg_by_bucket(edges_s: List[Edge]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    return:
      stance_indeg[cited_symbol]  = # of stance edges from session S
      neutral_indeg[cited_symbol] = # of value_neutral edges from session S
    """
    stance = defaultdict(int)
    neutral = defaultdict(int)

    for e in edges_s:
        if e.bucket == "stance":
            stance[e.cited] += 1
        elif e.bucket == "value_neutral":
            neutral[e.cited] += 1

    return stance, neutral


def node_type(sym: str, stance_cnt: int, neutral_cnt: int) -> str:
    """
    ノードを stance / neutral / mixed / none に分類
    - stance 優勢: stance/(stance+neutral) >= 0.6
    - neutral優勢: stance/(stance+neutral) <= 0.4
    - それ以外: mixed
    """
    tot = stance_cnt + neutral_cnt
    if tot == 0:
        return "none"
    ratio = stance_cnt / tot
    if ratio >= 0.6:
        return "stance"
    if ratio <= 0.4:
        return "value_neutral"
    return "mixed"


def prune_nodes_to_k(
    nodes: Set[str],
    edges_s: List[Edge],
    total_deg: Dict[str, int],
    k: int = 300
) -> Tuple[Set[str], List[Edge]]:
    """
    部分グラフの次数(in+out)上位kノードを残す。
    """
    if len(nodes) <= k:
        return nodes, edges_s

    keep = set([n for n, _ in sorted(total_deg.items(), key=lambda x: (-x[1], x[0]))[:k]])
    kept_edges = [e for e in edges_s if (e.citing in keep and e.cited in keep)]
    return keep, kept_edges


def scale_sizes(
    indeg: Dict[str, int],
    nodes: Set[str],
    min_size: float = 10.0,
    max_size: float = 100.0
) -> Dict[str, float]:
    vals = [indeg.get(n, 0) for n in nodes]
    vmax = max(vals) if vals else 0

    out: Dict[str, float] = {}
    if vmax <= 0:
        for n in nodes:
            out[n] = min_size
        return out

    for n in nodes:
        v = indeg.get(n, 0)
        out[n] = min_size + (max_size - min_size) * (v / vmax)
    return out


def edge_color_hex(e: Edge) -> str:
    if e.bucket == "value_neutral":
        return "#1f77b4"  # blue
    if e.bucket == "stance":
        return "#d62728"  # red
    return "#888888"      # gray


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ========= HTML (pyvis) =========
def build_pyvis_html(
    out_html: pathlib.Path,
    sess: int,
    nodes: Set[str],
    edges_s: List[Edge],
    titles: Dict[str, str],
    indeg_from_s: Dict[str, int],
    sizes: Dict[str, float],
    stance_indeg: Dict[str, int],     # ★ stance
    neutral_indeg: Dict[str, int],    # ★ value_neutral
) -> None:
    if not HAS_PYVIS:
        raise RuntimeError("pyvis is not installed. Please `pip install pyvis`.")

    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff")
    net.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=120, spring_strength=0.02, damping=0.09)

    for n in nodes:
        t = titles.get(n, "")
        deg = indeg_from_s.get(n, 0)

        s_cnt = stance_indeg.get(n, 0)
        neu_cnt = neutral_indeg.get(n, 0)
        ntype = node_type(n, s_cnt, neu_cnt)

        # node fill color (node-type based)
        if ntype == "stance":
            color = "#d62728"      # red
        elif ntype == "value_neutral":
            color = "#1f77b4"      # blue
        elif ntype == "mixed":
            color = "#9467bd"      # purple
        else:
            color = "#cccccc"      # gray

        # pyvis tooltip は HTML として解釈されがちなので、改行は &#10; が安全
        #（<br>でもよいが、文字列として出る環境があるため &#10; 推奨）
        parts = [n]
        if t:
            parts.append(t)
        parts.append(f"in-degree from session {sess}: {deg}")
        parts.append(f"STANCE={s_cnt}, NEUTRAL={neu_cnt}")
        tooltip = "&#10;".join(parts)

        net.add_node(
            n,
            label=n,
            title=tooltip,
            size=float(sizes.get(n, 10.0)),
            color=color,
        )

    for e in edges_s:
        if e.citing not in nodes or e.cited not in nodes:
            continue
        net.add_edge(e.citing, e.cited, color=edge_color_hex(e), title=e.relation)

    net.show_buttons(filter_=["physics"])
    net.write_html(str(out_html))

# ========= PNG (networkx + matplotlib) =========
def build_static_png(
    out_png: pathlib.Path,
    sess: int,
    nodes: Set[str],
    edges_s: List[Edge],
    titles: Dict[str, str],
    indeg_from_s: Dict[str, int],
    sizes: Dict[str, float],
    seed: int = 42,
) -> None:
    if not HAS_NX:
        raise RuntimeError("networkx/matplotlib not installed. Please `pip install networkx matplotlib`.")

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)

    edge_colors = []
    for e in edges_s:
        if e.citing in nodes and e.cited in nodes:
            G.add_edge(e.citing, e.cited, bucket=e.bucket)
            edge_colors.append(edge_color_hex(e))

    # レイアウト（ノード数300前後ならこれで現実的）
    # iterations を上げると見栄えは良くなるが重くなる
    pos = nx.spring_layout(G, seed=seed, k=None, iterations=80)

    # node sizes: networkx expects list aligned with nodelist
    nodelist = list(G.nodes())
    node_sizes = [float(sizes.get(n, 10.0)) for n in nodelist]

    plt.figure(figsize=(16, 10))

    # nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_sizes, alpha=0.85)

    # edges（色は e.bucket に対応させたいので、edgelist順に作る）
    edgelist = list(G.edges())
    # 上でedge_colorsは edges_s の順に積んだが、G.edges順と一致しない可能性があるので作り直す
    ecolors = []
    for u, v in edgelist:
        b = G[u][v].get("bucket")
        if b == "weak":
            ecolors.append("#1f77b4")
        elif b == "strong":
            ecolors.append("#d62728")
        else:
            ecolors.append("#888888")

    nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=ecolors, arrows=True, alpha=0.55, width=1.0)

    # ラベルは重いので「Top10 cited」だけに絞る（見やすさ・軽さ優先）
    top_nodes = sorted([(n, indeg_from_s.get(n, 0)) for n in nodes], key=lambda x: (-x[1], x[0]))[:10]
    label_nodes = {n for n, _ in top_nodes if indeg_from_s.get(n, 0) > 0}

    labels = {}
    for n in label_nodes:
        t = titles.get(n, "")
        # PNGは文字が潰れやすいので symbol のみ（必要なら title を短縮して併記に変更可）
        labels[n] = n

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(f"UNGA Resolution Citation Graph (Session {sess})\nnode size ~ in-degree from session {sess} (unspecified excluded)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", type=int, required=True, help="latest session (e.g., 79)")
    ap.add_argument("--N", type=int, required=True, help="how many sessions to include (e.g., 3 => X-2..X)")
    ap.add_argument("--k", type=int, default=300, help="target nodes to keep for graph (default 300)")
    ap.add_argument("--no-graph", action="store_true", help="skip html graph output")
    ap.add_argument("--no-png", action="store_true", help="skip png output")
    args = ap.parse_args()

    if not EDGES_CSV.exists():
        raise FileNotFoundError(EDGES_CSV)

    titles = load_titles(TITLES_TSV)
    edges_all = list(iter_edges(EDGES_CSV))  # unspecified除外済

    sessions = list(range(args.X - args.N + 1, args.X + 1))
    ensure_dir(OUT_DIR)
    summary = {
        "X": args.X,
        "N": args.N,
        "sessions": sessions,
        "k_nodes_target": args.k,
        "excluded_relations": sorted(EXCLUDE_RELATIONS),
        "relation_groups": ["value_neutral", "stance", "other"],
        "outputs": [],
    }

    if (not args.no_graph) and (not HAS_PYVIS):
        print("NOTE: pyvis not installed; HTML graphs will be skipped. Install: pip install pyvis")
        args.no_graph = True

    if (not args.no_png) and (not HAS_NX):
        print("NOTE: networkx/matplotlib not installed; PNG graphs will be skipped. Install: pip install networkx matplotlib")
        args.no_png = True

    for sess in sessions:
        out_sess = OUT_DIR / f"session_{sess}"
        ensure_dir(out_sess)

        edges_s = session_edges(edges_all, sess)
        stance_indeg, neutral_indeg = split_indeg_by_bucket(edges_s)
        # 1-2) Top10 (a/b/c)
        top_all   = top10_in_degree(edges_s, titles, "all")
        top_neu   = top10_in_degree(edges_s, titles, "value_neutral")
        top_sta   = top10_in_degree(edges_s, titles, "stance")

        write_top10_tsv(out_sess / "top10_in_degree_all.tsv", top_all)
        write_top10_tsv(out_sess / "top10_in_degree_neutral.tsv", top_neu)
        write_top10_tsv(out_sess / "top10_in_degree_stance.tsv", top_sta)

        # 3) Subgraph & prune
        nodes, edges_sub, indeg_from_s, total_deg = build_subgraph(edges_s)
        kept_nodes, kept_edges = prune_nodes_to_k(nodes, edges_sub, total_deg, k=args.k)

        sizes = scale_sizes(indeg_from_s, kept_nodes, min_size=10.0, max_size=100.0)

        # outputs
        graph_html = out_sess / "graph.html"
        graph_png = out_sess / "graph.png"

        if not args.no_graph:
            build_pyvis_html(
                graph_html,
                sess=sess,
                nodes=kept_nodes,
                edges_s=kept_edges,
                titles=titles,
                indeg_from_s=indeg_from_s,
                sizes=sizes,
                stance_indeg=stance_indeg,
                neutral_indeg=neutral_indeg,
            )

        if not args.no_png:
            build_static_png(
                graph_png,
                sess=sess,
                nodes=kept_nodes,
                edges_s=kept_edges,
                titles=titles,
                indeg_from_s=indeg_from_s,
                sizes=sizes,
                seed=42,
            )

        meta = {
            "session": sess,
            "edges_in_session_excl_unspecified": len(edges_s),
            "seed_nodes_citing_in_session": len({e.citing for e in edges_s}),
            "C(S)_unique_cited": len({e.cited for e in edges_s}),
            "nodes_in_subgraph_before_prune": len(nodes),
            "nodes_kept": len(kept_nodes),
            "edges_kept": len(kept_edges),
            "outputs": {
                "top10_all": str(out_sess / "top10_in_degree_all.tsv"),
                "top10_weak": str(out_sess / "top10_in_degree_weak.tsv"),
                "top10_strong": str(out_sess / "top10_in_degree_strong.tsv"),
                "graph_html": str(graph_html) if (not args.no_graph) else None,
                "graph_png": str(graph_png) if (not args.no_png) else None,
            },
        }

        (out_sess / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["outputs"].append(meta)

        print(f"[session {sess}] edges={len(edges_s)} nodes_before={len(nodes)} kept_nodes={len(kept_nodes)} kept_edges={len(kept_edges)}")

    (OUT_DIR / "trend_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done")
    print("output:", OUT_DIR)

if __name__ == "__main__":
    main()