#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import pathlib
import re
import requests
from bs4 import BeautifulSoup

OUTDIR = pathlib.Path("ga_symbols_from_research_un")
OUTDIR.mkdir(exist_ok=True)

SYMBOLS_PATH = OUTDIR / "symbols.txt"     # 1行1symbol（追記）
STATE_PATH   = OUTDIR / "state.json"      # 再開用
ERRORS_PATH  = OUTDIR / "errors.jsonl"    # エラー記録

BASE = "https://research.un.org/en/docs/ga/quick/regular"

# まずは現実的に 1〜80期。必要なら増やせます。
MAX_SESSION = 80

# リトライ対象
RETRY_STATUSES = {429, 500, 502, 503, 504}

UA = {"User-Agent": "GA-RES-Collector/3.0"}

# A/RES/103(I) や A/RES/78/103 などを拾う
SYM_RE = re.compile(r"\bA/RES/\d+(?:/\d+|\([IVXLCDM]+\))\b", re.IGNORECASE)

def append_line(path: pathlib.Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def append_jsonl(path: pathlib.Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"session": 1}

def save_state(st: dict) -> None:
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

def load_symbols_set() -> set[str]:
    if not SYMBOLS_PATH.exists():
        return set()
    return set(x.strip() for x in SYMBOLS_PATH.read_text(encoding="utf-8").splitlines() if x.strip())

def fetch_html(url: str, max_retries: int = 8) -> str:
    backoff = 1.0
    for _ in range(max_retries):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200 and r.text:
            return r.text
        if r.status_code in RETRY_STATUSES:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        # 404なら会期ページが無い（上限超えなど）
        if r.status_code == 404:
            return ""
        r.raise_for_status()
    raise RuntimeError(f"fetch failed: {url} last_status={r.status_code}")

def extract_symbols_from_page(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    # テーブル全体のテキストから A/RES/... パターンを拾う（構造が変わっても強い）
    text = soup.get_text(" ", strip=True)
    syms = set(m.group(0).upper() for m in SYM_RE.finditer(text))

    # 念のため “決議”以外の A/RES/xx/(I) みたいな揺れが入る場合もあるので、
    # ここではA/RESで始まるものだけ残す（上で保証されているが保険）
    syms = {s for s in syms if s.startswith("A/RES/")}

    # 並びは一旦ソート（後で年や番号順に並べたければ別工程でOK）
    return sorted(syms)

def main():
    st = load_state()
    start_sess = int(st.get("session", 1))

    seen = load_symbols_set()
    total_new = 0

    for sess in range(start_sess, MAX_SESSION + 1):
        url = f"{BASE}/{sess}"
        html = fetch_html(url)

        if not html:
            # 会期ページ自体が無いなら、ここで止めてもよい
            append_jsonl(ERRORS_PATH, {"session": sess, "url": url, "note": "no page (404 or empty)"})
            save_state({"session": sess + 1})
            print(f"session {sess}: page missing/empty")
            continue

        syms = extract_symbols_from_page(html)

        # その会期のページにある A/RES のみを追記
        new = [s for s in syms if s not in seen]
        for s in new:
            append_line(SYMBOLS_PATH, s)
            seen.add(s)

        total_new += len(new)
        print(f"session {sess}: found {len(syms)} symbols, new {len(new)}")
        save_state({"session": sess + 1})
        time.sleep(0.3)

    print("done")
    print("total symbols:", len(seen), "new added:", total_new)
    print("saved:", SYMBOLS_PATH)

if __name__ == "__main__":
    main()