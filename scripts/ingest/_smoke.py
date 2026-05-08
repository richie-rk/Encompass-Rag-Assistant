"""Smoke-test the OAS synthesis on a handful of sentinel JSONL records.

Run from the repo root:  python -m scripts.ingest._smoke
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Re-encode stdout as UTF-8 on Windows so we don't crash on non-ASCII chars.
if hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from scripts.ingest.oas_synth import synthesize_endpoint_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONL_PATH = REPO_ROOT / "scripts" / "data" / "developer_connect.jsonl"

SENTINELS = [
    "get-borrower-pairs",
    "assign-document-attachments",
    "create-loan",
    "get-loan",
    "view-pipeline",
]


def load_records(slugs):
    wanted = set(slugs)
    found = {}
    with JSONL_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("slug") in wanted:
                found.setdefault(r["slug"], r)
                if len(found) == len(wanted):
                    break
    return found


def main() -> int:
    recs = load_records(SENTINELS)
    failures = []
    for slug in SENTINELS:
        rec = recs.get(slug)
        if rec is None:
            print(f"=== {slug}: NOT FOUND in jsonl ===")
            failures.append(slug)
            continue
        md = synthesize_endpoint_markdown(rec)
        body = rec.get("body_md") or ""
        oas_paths = len((rec.get("oas") or {}).get("paths") or {})
        doc_api_chars = len(json.dumps(rec.get("doc_api") or {}, ensure_ascii=False))
        print(f"=== {slug} ===")
        print(f"  body_md_chars={len(body)}  oas paths={oas_paths}  doc_api_chars={doc_api_chars}")
        print(f"  synthesized markdown chars: {len(md):,}")
        if md:
            preview = "\n".join(md.splitlines()[:30])
            print("  --- first 30 lines ---")
            for line in preview.splitlines():
                print(f"  {line}")
            print("  --- end preview ---")
        else:
            print("  (no markdown produced)")
        print()
        if len(md) < 500:
            failures.append(slug)

    if failures:
        print(f"FAIL: {len(failures)} sentinel(s) under threshold or missing: {failures}")
        return 1
    print("PASS: all sentinels produced >=500 chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
