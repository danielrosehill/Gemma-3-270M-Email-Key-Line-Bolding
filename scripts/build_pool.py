"""Build pool/pool.jsonl: the synthetic dictated emails from ASR-Paragraph-Breaks, split into sentences.

  python3 scripts/build_pool.py            # writes data/pool/pool.jsonl
  python3 scripts/build_pool.py --upload   # also pushes it to the private HF dataset (needs HF_TOKEN)

Source: the sibling repo's generated memos, data/synthetic/synthetic-*.jsonl (gitignored there, so a
local clone at $SIBLING, default ../ASR-Paragraph-Breaks, with those files is needed). Kept: spec.kind
starting "a dictated email to" (43 from seed 1, 299 from the email-only seed 3; 342 on 2026-10-21).
Excluded: "a dictated complaint to a company's customer service" (24): those read as phone calls or
voicemails ("Hi, this is Dana calling about..."), not emails. Messages to family/friends are not emails.

Paragraphing: Daniel's hand-reviewed version from the danielrosehill/ASR-Paragraph-Breaks annotations
where one exists, else Haiku's proposal plus style_rules.py (the same rules that dataset uses).
Sentences come from style_rules.sentences (full-stop split with an abbreviation guard).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from style_rules import apply as style_rules, sentences

ROOT = Path(__file__).resolve().parent.parent
SIBLING = Path(os.environ.get("SIBLING", ROOT.parent / "ASR-Paragraph-Breaks"))
OUT = ROOT / "data" / "pool" / "pool.jsonl"
DATASET = "danielrosehill/Email-Key-Line-Bolding"
PARA_DATASET = "danielrosehill/ASR-Paragraph-Breaks"
# Seed 1 had two email kinds; seed 3 (2026-10-21, 299 emails) used the 12 EMAIL_KINDS in the sibling's
# generate_synthetic.py. All of them, and only them, start with this prefix.
EMAIL_PREFIX = "a dictated email to"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def gold_paragraphs() -> dict[str, list[str]]:
    tree = json.loads(fetch(f"https://huggingface.co/api/datasets/{PARA_DATASET}/tree/main/annotations"))
    out = {}
    for f in tree:
        if f["path"].endswith(".jsonl"):
            r = json.loads(fetch(f"https://huggingface.co/datasets/{PARA_DATASET}/resolve/main/{f['path']}"))
            out[r["id"]] = r["paragraphs"]
    return out


def main() -> None:
    gold = gold_paragraphs()
    rows = []
    for f in sorted((SIBLING / "data" / "synthetic").glob("synthetic-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            spec = r["spec"]
            if not spec["kind"].startswith(EMAIL_PREFIX):
                continue
            paras = gold.get(r["id"]) or style_rules(r["proposed"])
            rows.append({
                "id": r["id"], "source_id": r["id"], "kind": spec["kind"], "topic": spec["topic"], "style": spec["style"],
                "paragraph_source": "gold" if r["id"] in gold else "style_rules v1.1",
                "paragraphs": paras, "sentences": [sentences(p) for p in paras],
                "words": sum(len(p.split()) for p in paras), "generator": r["model"],
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_gold = sum(r["paragraph_source"] == "gold" for r in rows)
    print(f"{len(rows)} emails ({n_gold} with hand-reviewed paragraphs) -> {OUT.relative_to(ROOT)}")

    if "--upload" in sys.argv:
        env = {**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN") or os.environ["HUGGINGFACE_TOKEN"]}
        subprocess.run([os.path.expanduser("~/.local/bin/hf"), "upload", DATASET, str(OUT), "pool/pool.jsonl",
                        "--repo-type", "dataset", "--commit-message", f"Rebuild pool: {len(rows)} emails"],
                       check=True, env=env)


if __name__ == "__main__":
    main()
