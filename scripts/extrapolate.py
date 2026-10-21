# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "huggingface_hub"]
# ///
"""Extrapolate Daniel's bolding to the unannotated pool with an LLM given the written rules.

  uv run scripts/extrapolate.py eval [--model claude-haiku-5-5] [--folds 4]  # k-fold over gold: word P/R/F1
  uv run scripts/extrapolate.py label [--model ...]                          # silver for the rest of the pool

The model gets docs/bolding-style.md (as SYSTEM below), gold annotations as worked examples and the
email; it returns the email with ** inserted. Output is accepted only if removing ** gives the text
back exactly; spans are then snapped outward to whole words and clipped to one paragraph, the same
normalisation the annotator applies. Rejected outputs are retried once, then dropped.

Scoring is per word: a word is "bold" if it lies inside a span. Gold read from the HF dataset
(data/hf/annotations after a snapshot_download, see main()). Test ids (TEST_IDS) are never used as
examples for `label`, so the trained model's test set stays clean. Results are in docs/bolding-style.md.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DATASET = "danielrosehill/Email-Key-Line-Bolding"
HF = ROOT / "data" / "hf"
SILVER = ROOT / "data" / "silver.jsonl"
WORD = re.compile(r"[\w'’]")

# Held out for the trained model: 6 of the first 19 gold (2026-10-21), picked to cover short and long emails
# and five recipient kinds; see build_training.py.
TEST_IDS = {"syn-3-0027", "syn-3-0078", "syn-3-0090", "syn-3-0130", "syn-3-0170", "syn-3-0272"}

SYSTEM = """You mark the key parts of a dictated email in Markdown bold, following one person's rules exactly.

The goal: someone skimming only the bold should know what the email is about and what is being asked of them.

Bold:
1. The ask. Every direct request or question to the recipient ("Could you...", "Can you confirm...", "Please send...", "Let me know by..."). This is the most consistently bolded thing.
2. The purpose line, when the email states why it is written ("I'm writing to...", "I wanted to let you know..."), unless it is just the greeting.
3. Each item of a list of points or steps: the sentence that carries "First," / "Second," / "Third," / "Also," / "Finally," and its point.
4. The key facts the recipient must act on or remember: the decisive date, time, place, amount, deadline, or the problem itself ("still hasn't arrived", "the hydraulic arm is leaking").
5. Instructions and warnings ("Don't use it until...", "Please don't proceed above £250 without contacting me").

Do not bold:
- The greeting or opening line ("Hi Mr. Okonkwo, this is Priya...", "Dear members,"). The first paragraph is almost never bold unless it is itself the request.
- Sign-offs, thanks and pleasantries ("Thanks so much", "Looking forward to hearing from you", names).
- Background, explanation, apologies, reasons and asides that support a point rather than make it.
- Routine contact details, unless they are part of the ask.

Form:
- A span is usually one whole sentence. Bold a clause or phrase instead when only part of the sentence carries the point (drop a trailing "because..." or a lead-in like "so"). Occasionally two consecutive sentences that form one request go in one span.
- Never bold across a paragraph break. Never bold part of a word.
- Density: about a quarter of the words, typically 2 to 6 spans, roughly one span per 80 words. A short, simple email may need only one or two. If nothing is worth bolding, return the email unchanged.

Only insert ** markers. Never change, add or remove any other character, including whitespace and paragraph breaks. Output only the email."""


def api_key() -> str:
    # The shell's ANTHROPIC_API_KEY was invalid on 2026-10-21, hence the check and the 1Password fallback
    # (same as ASR-Paragraph-Breaks scripts/generate_synthetic.py).
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        try:
            anthropic.Anthropic(api_key=key).models.retrieve("claude-haiku-5-5")
            return key
        except anthropic.AuthenticationError:
            pass
    return subprocess.run(
        ["op", "item", "get", "Anthropic / Claude Experiments API Key", "--vault", "API Keys & Programmatic",
         "--fields", "password", "--reveal"], check=True, capture_output=True, text=True).stdout.strip()


def is_word(c: str) -> bool:
    return bool(c) and bool(WORD.match(c))


def gold() -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted((HF / "annotations").glob("*.jsonl"))]
    return sorted(rows, key=lambda r: r["id"])


def pool() -> list[dict]:
    return [json.loads(l) for l in (HF / "pool" / "pool.jsonl").read_text().splitlines() if l.strip()]


def text_of(item: dict) -> str:
    return "\n\n".join(item["paragraphs"])


def bolded(text: str, spans: list[list[int]]) -> str:
    out, pos = [], 0
    for a, b in spans:
        out += [text[pos:a], "**", text[a:b], "**"]
        pos = b
    return "".join(out + [text[pos:]])


def parse(text: str, out: str) -> list[list[int]] | None:
    """Model output -> normalised spans, or None if anything but ** changed."""
    out = out.strip()
    if out.replace("**", "") != text:
        return None
    spans, pos, inside, start = [], 0, False, 0
    for piece in re.split(r"(\*\*)", out):
        if piece == "**":
            if inside:
                spans.append([start, pos])
            else:
                start = pos
            inside = not inside
        else:
            pos += len(piece)
    if inside:
        return None
    norm = []
    for a, b in spans:
        # clip to one paragraph, trim whitespace and edge punctuation-only runs, snap to whole words
        para_end = text.find("\n\n", a)
        if para_end != -1 and para_end < b:
            b = para_end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        while a > 0 and is_word(text[a - 1]) and is_word(text[a]):
            a -= 1
        while b < len(text) and is_word(text[b - 1]) and is_word(text[b]):
            b += 1
        if b > a:
            norm.append([a, b])
    norm.sort()
    merged: list[list[int]] = []
    for a, b in norm:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def word_labels(text: str, spans: list[list[int]]) -> list[bool]:
    return [any(a <= m.start() < b for a, b in spans) for m in re.finditer(r"\S+", text)]


def params(model: str, examples: list[dict], text: str) -> dict:
    msgs = []
    for e in examples:
        msgs += [{"role": "user", "content": e["text"]}, {"role": "assistant", "content": e["bolded"]}]
    msgs.append({"role": "user", "content": text})
    return dict(model=model, max_tokens=4000, system=SYSTEM, messages=msgs, cache_control={"type": "ephemeral"})


def run(client, model: str, examples: list[dict], text: str) -> list[list[int]] | None:
    for _ in range(2):
        msg = client.messages.create(**params(model, examples, text))
        spans = parse(text, next((b.text for b in msg.content if b.type == "text"), ""))
        if spans is not None:
            return spans
    return None


def evaluate(model: str, folds: int) -> None:
    g = gold()
    random.Random(0).shuffle(g)
    client = anthropic.Anthropic(api_key=api_key())
    jobs = []
    for k in range(folds):
        test = g[k::folds]
        jobs += [(item, [x for x in g if x not in test]) for item in test]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda j: run(client, model, j[1], j[0]["text"]), jobs))
    tp = npred = nref = rejected = 0
    per = []
    for (item, _), spans in zip(jobs, outs):
        if spans is None:
            rejected += 1
            spans = []
        p, r = word_labels(item["text"], spans), word_labels(item["text"], item["bold_spans"])
        t = sum(a and b for a, b in zip(p, r))
        tp, npred, nref = tp + t, npred + sum(p), nref + sum(r)
        f = 2 * t / (sum(p) + sum(r)) if sum(p) + sum(r) else 1.0
        per.append((item["id"], len(spans), item["n_bold"], round(f, 2)))
    P, R = tp / npred, tp / nref
    print(f"{model}: {len(g)} gold, {folds}-fold, {rejected} rejected")
    print(f"  word P {P:.2f}  R {R:.2f}  F1 {2 * P * R / (P + R):.2f}  (pred bold {npred}, gold bold {nref})")
    for row in sorted(per):
        print("  ", *row)


def label(model: str) -> None:
    g = gold()
    examples = [x for x in g if x["id"] not in TEST_IDS]
    done = {x["id"] for x in g}
    prev = {}
    if SILVER.exists():
        prev = {r["id"]: r for r in map(json.loads, SILVER.read_text().splitlines())}
    todo = [p for p in pool() if p["id"] not in done and p["id"] not in prev]
    client = anthropic.Anthropic(api_key=api_key())
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda p: run(client, model, examples, text_of(p)), todo))
    dropped = 0
    for p, spans in zip(todo, outs):
        if spans is None:
            dropped += 1
            continue
        t = text_of(p)
        prev[p["id"]] = {"id": p["id"], "kind": p["kind"], "text": t, "bold_spans": spans,
                         "bolded": bolded(t, spans), "labeller": model}
    SILVER.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(prev.values(), key=lambda r: r["id"])))
    print(f"{len(prev)} silver in {SILVER.relative_to(ROOT)} ({len(todo) - dropped} new, {dropped} dropped), "
          f"{len(examples)} gold examples")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    model = args.get("--model", "claude-haiku-5-5")
    if cmd not in ("eval", "label"):
        sys.exit(__doc__)
    snapshot_download(DATASET, repo_type="dataset", local_dir=str(HF))
    if cmd == "eval":
        evaluate(model, int(args.get("--folds", 4)))
    else:
        label(model)


if __name__ == "__main__":
    main()
