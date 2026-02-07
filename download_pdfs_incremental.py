#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import time
import re
import requests

# ---- inputs ----
SYMBOLS_TXT = pathlib.Path("undl_dump") / "symbols.txt"
IN_JSONL    = pathlib.Path("undl_dump") / "records.jsonl"      # 既存（主に76期まで）
DONE        = pathlib.Path("undl_dump") / "done_pdf_symbols.txt"
FAILS_JSONL = pathlib.Path("undl_dump") / "fails.jsonl"        # 過去失敗（今回は除外）

# ---- outputs ----
PDF_DIR = pathlib.Path("undl_dump") / "pdf_en"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---- network ----
UA = "GA-RES-PDF-Downloader/3.0"
TIMEOUT = 120
SLEEP = 0.2

ODS_API = "https://documents.un.org/api/symbol/access"  # 公式ODS

# symbolパターン
# A/RES/78/123  / A/RES/14(I) / A/RES/77/1A など
RE_SYM = re.compile(r"^A/RES/\d+/(?:\d{1,4}[A-Z]?|\d+\([IVXLCDM]+\))$", re.I)

# ★前方一致事故が多い: 決議番号が 1〜2桁（+ 1A 等）だけを判定
RE_LOW_NO = re.compile(r"^A/RES/\d+/(\d{1,2})([A-Z]?)$", re.I)

# recjson中に symbol が完全一致で含まれているかの緩い検証（安全側）
def recjson_contains_symbol(recjson_obj, sym: str) -> bool:
    try:
        blob = json.dumps(recjson_obj, ensure_ascii=False)
    except Exception:
        return False
    return sym.upper() in blob.upper()

def load_set(p: pathlib.Path) -> set[str]:
    if not p.exists():
        return set()
    return set(x.strip().upper() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip())

def append_line(p: pathlib.Path, s: str) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(s + "\n")

def sym_to_filename(sym: str) -> str:
    return sym.replace("/", "_").replace("(", "").replace(")", "") + ".pdf"

def pick_english_pdf_url(record_obj) -> str | None:
    """
    record_obj は records.jsonl の obj["record"] を想定（list or dict）。
    """
    if isinstance(record_obj, list) and record_obj:
        rec = record_obj[0]
    else:
        rec = record_obj

    files = None
    if isinstance(rec, dict):
        bib = rec.get("BIB")
        if isinstance(bib, dict) and isinstance(bib.get("files"), list):
            files = bib["files"]
        elif isinstance(rec.get("files"), list):
            files = rec["files"]

    if not files:
        return None

    # description "English"
    for f in files:
        if not isinstance(f, dict):
            continue
        if (f.get("description") or "").lower() == "english" and f.get("url"):
            return f["url"]

    # fallback: -EN.pdf
    for f in files:
        if not isinstance(f, dict):
            continue
        url = f.get("url") or ""
        name = f.get("full_name") or ""
        if ("-EN.pdf" in name) or ("-EN.pdf" in url):
            if f.get("url"):
                return f["url"]

    return None

def build_symbol_to_record_from_records() -> dict[str, dict]:
    """
    records.jsonl から symbol -> record_obj を構築。
    ※ URLだけでなく record 全体を保持して、後で “symbol一致検証” に使う。
    """
    mp: dict[str, dict] = {}
    if not IN_JSONL.exists():
        return mp

    with IN_JSONL.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            sym = (obj.get("symbol") or "").strip().upper()
            if not sym:
                continue
            rec = obj.get("record")
            if rec is None:
                continue

            # 先勝ち（後勝ちにしたいなら mp[sym]=... にする）
            mp.setdefault(sym, rec)

    return mp

def load_failed_symbols(path: pathlib.Path) -> set[str]:
    failed: set[str] = set()
    if not path.exists():
        return failed

    keys = ("symbol", "sym", "s")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                for k in keys:
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        failed.add(v.strip().upper())
                        break
            elif isinstance(obj, str) and obj.strip():
                failed.add(obj.strip().upper())
    return failed

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def download_via_undl(session: requests.Session, sym: str, record_obj, out: pathlib.Path) -> bool:
    """
    records.jsonl 由来の record_obj から英語PDF URLを取り、DL。
    ただし、record_obj が sym と一致しているか簡易検証を入れる。
    """
    # ★symbol完全一致チェック（誤マッピング対策）
    if not recjson_contains_symbol(record_obj, sym):
        print("skip(undl-mismatch)", sym, "record does not contain symbol; fallback to ODS")
        return False

    url = pick_english_pdf_url(record_obj)
    if not url:
        print("skip(undl-no-en-url)", sym, "fallback to ODS")
        return False

    r = session.get(url, timeout=TIMEOUT)
    if r.status_code != 200 or not r.content:
        print("fail(undl)", sym, r.status_code, url)
        return False

    # ざっくりPDF検証
    if not r.content.startswith(b"%PDF"):
        print("fail(undl-not-pdf)", sym, "len", len(r.content), url)
        return False

    out.write_bytes(r.content)
    return True

def download_via_ods(session: requests.Session, sym: str, out: pathlib.Path) -> bool:
    params = {"l": "en", "s": sym, "t": "pdf"}
    r = session.get(ODS_API, params=params, timeout=TIMEOUT, allow_redirects=True)
    if r.status_code != 200 or not r.content:
        print("fail(ods)", sym, r.status_code, r.url)
        return False
    if not r.content.startswith(b"%PDF"):
        print("fail(ods-not-pdf)", sym, "len", len(r.content), r.url)
        return False
    out.write_bytes(r.content)
    return True

def should_force_redownload(sym: str, out: pathlib.Path) -> bool:
    """
    ★運用オプション：
    - 2桁以下の決議（前方一致事故ゾーン）は、既存PDFがあっても再DLして上書きする
      （本当に取り違えが起きたのがここなので）
    """
    if RE_LOW_NO.match(sym):
        return True
    return False

def main():
    if not SYMBOLS_TXT.exists():
        raise FileNotFoundError(SYMBOLS_TXT)

    done = load_set(DONE)
    failed = load_failed_symbols(FAILS_JSONL)

    # ★差分DL：symbols.txt のうち、doneにもfailedにも無いものだけ
    all_syms = [ln.strip().upper() for ln in SYMBOLS_TXT.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    all_syms = [s for s in all_syms if RE_SYM.match(s)]

    targets = [s for s in all_syms if (s not in done) and (s not in failed)]

    print(
        "symbols total:", len(all_syms),
        "done:", len(done),
        "failed(skip):", len(failed),
        "to_download(diff):", len(targets),
    )

    # records.jsonl 由来の map（あればUNDL経由で高速）
    sym2record = build_symbol_to_record_from_records()

    s = requests.Session()
    s.headers.update({"User-Agent": UA})

    ok = 0
    for i, sym in enumerate(targets, 1):
        out = PDF_DIR / sym_to_filename(sym)

        # すでにPDFがある場合でも、2桁以下は取り違え修復のため再DL推奨
        if out.exists() and out.stat().st_size > 1000 and not should_force_redownload(sym, out):
            append_line(DONE, sym)
            done.add(sym)
            ok += 1
            continue

        # 優先順：
        # 1) records.jsonl に symbol の record があるなら UNDL（ただし “symbol一致検証”あり）
        # 2) それ以外は ODS
        rec = sym2record.get(sym)
        success = False

        if rec is not None:
            success = download_via_undl(s, sym, rec, out)

        if not success:
            success = download_via_ods(s, sym, out)

        if success:
            append_line(DONE, sym)
            done.add(sym)
            ok += 1
            if ok <= 10 or ok % 200 == 0:
                print("ok", ok, sym)
        else:
            # 失敗はdoneにもfailsにも追加しない（運用は別で）
            time.sleep(1.5)

        time.sleep(SLEEP)

    print("done downloaded:", ok)

if __name__ == "__main__":
    main()