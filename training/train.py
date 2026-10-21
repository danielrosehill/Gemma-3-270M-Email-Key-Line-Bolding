"""Fine-tune: paragraphed email -> the same email with the key parts in **bold**.

Runs on Modal's `ai-training` app (ai-models/Modal-AI-Training); command in docs/training.md.
Expects train.jsonl / test.jsonl next to this file (scripts/build_training.py) and writes to $OUT:
the model, test predictions from the base and fine-tuned model, and metrics.json.

Scoring: an output that is not the input plus ** markers is a "fallback" (the input is used unbolded,
as it would be in deployment). Bold is compared per word (a word is bold if it starts inside a span),
the same measure as scripts/extrapolate.py, so the Sonnet labeller's cross-validated F1 is comparable.
Template: ASR-Paragraph-Breaks training/train.py (same image, same TRL setup).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import transformers
import trl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = os.environ.get("BASE", "google/gemma-3-270m-it")
OUT = Path(os.environ.get("OUT", "out"))
HERE = Path(__file__).parent
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "5e-5"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "0"))  # smoke runs: cap the train set

SYSTEM = ("Put the key parts of this email in Markdown bold (**like this**): the requests, the purpose, "
          "the main points and the key facts. Do not change, add or remove anything else.")


def spans(marked):
    out, pos, start, inside = [], 0, 0, False
    for piece in re.split(r"(\*\*)", marked):
        if piece == "**":
            if inside:
                out.append((start, pos))
            else:
                start = pos
            inside = not inside
        else:
            pos += len(piece)
    return out


def word_labels(text, sp):
    return [any(a <= m.start() < b for a, b in sp) for m in re.finditer(r"\S+", text)]


def load(name):
    return [json.loads(l) for l in open(HERE / name)]


def prompt(tok, r):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"]}],
                                   tokenize=False, add_generation_prompt=True)


def generate(model, tok, rows):
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        enc = tok([prompt(tok, r) for r in batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1600, do_sample=False)
        for r, o in zip(batch, out):
            preds.append({"id": r["id"], "pred": tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()})
    return preds


def score(rows, preds):
    by = {p["id"]: p["pred"] for p in preds}
    tp = npred = nref = fallback = 0
    for r in rows:
        pred = by[r["id"]]
        if pred.replace("**", "") != r["input"].strip() or pred.count("**") % 2:
            fallback += 1
            pred = r["input"]
        p, g = word_labels(r["input"], spans(pred)), word_labels(r["input"], spans(r["output"]))
        tp, npred, nref = tp + sum(a and b for a, b in zip(p, g)), npred + sum(p), nref + sum(g)
    p, rc = (tp / npred if npred else 0.0), (tp / nref if nref else 0.0)
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(2 * p * rc / (p + rc) if p + rc else 0.0, 3),
            "fallback_rate": round(fallback / len(rows), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"transformers {transformers.__version__}, trl {trl.__version__}", file=sys.stderr)
    train, test = load("train.jsonl"), load("test.jsonl")
    if MAX_TRAIN:
        train, test = train[:MAX_TRAIN], test[:4]
    tok = AutoTokenizer.from_pretrained(BASE)
    t0 = time.time()

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    pred_base = generate(base, tok, test)
    (OUT / "pred_base.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pred_base))
    del base
    torch.cuda.empty_cache()
    t_base = time.time() - t0

    # Plain-string prompt/completion: TRL trains on the completion only.
    ds = Dataset.from_list([{"prompt": prompt(tok, r), "completion": r["output"] + tok.eos_token} for r in train])
    cfg = SFTConfig(output_dir=str(OUT / "ckpt"), num_train_epochs=EPOCHS, learning_rate=LR,
                    per_device_train_batch_size=4, gradient_accumulation_steps=2, gradient_checkpointing=True,
                    bf16=True, max_length=4096, logging_steps=10, save_strategy="no", report_to=[],
                    lr_scheduler_type="cosine", warmup_steps=10)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    t1 = time.time()
    trainer.train()
    t_train = time.time() - t1
    trainer.save_model(str(OUT / "model"))
    tok.save_pretrained(str(OUT / "model"))

    t2 = time.time()
    pred_ft = generate(trainer.model, tok, test)
    (OUT / "pred_ft.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pred_ft))
    metrics = {"base": BASE, "epochs": EPOCHS, "lr": LR, "n_train": len(train), "n_test": len(test),
               "base_model": score(test, pred_base), "fine_tuned": score(test, pred_ft),
               "seconds": {"base_eval": round(t_base), "train": round(t_train), "ft_eval": round(time.time() - t2)},
               "log": trainer.state.log_history}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("base_model", "fine_tuned", "seconds")}), file=sys.stderr)


if __name__ == "__main__":
    main()
