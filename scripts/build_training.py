"""Build training/train.jsonl and training/test.jsonl.

  uv run scripts/extrapolate.py label     # first: data/silver.jsonl
  uv run --with anthropic --with huggingface_hub scripts/build_training.py

Test: the gold annotations in extrapolate.TEST_IDS (never used as LLM examples). Train: the other gold
plus every silver row (Sonnet labelling of the rest of the pool). Each row: {id, label, input, output},
input the paragraphed email, output the same text with ** around the bold spans.
Gold is read from data/hf/annotations, which extrapolate.py refreshes from HF on every run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training"

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extrapolate import SILVER, TEST_IDS, gold  # noqa: E402


def main() -> None:
    train, test = [], []
    g = gold()
    for r in g:
        row = {"id": r["id"], "label": "gold", "input": r["text"], "output": r["bolded"]}
        (test if r["id"] in TEST_IDS else train).append(row)
    gold_ids = {r["id"] for r in g}
    for line in SILVER.read_text().splitlines():
        r = json.loads(line)
        if r["id"] not in gold_ids:
            train.append({"id": r["id"], "label": "silver", "input": r["text"], "output": r["bolded"]})
    for row in train + test:
        assert row["output"].replace("**", "") == row["input"], row["id"]
    random.Random(0).shuffle(train)
    test.sort(key=lambda r: r["id"])
    OUT.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_gold = sum(r["label"] == "gold" for r in train)
    print(f"train {len(train)} ({n_gold} gold, {len(train) - n_gold} silver), test {len(test)} gold")


if __name__ == "__main__":
    main()
