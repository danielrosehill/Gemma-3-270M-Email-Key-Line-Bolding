# Training

Runs go to the Modal app `ai-training` (`~/repos/github/ai-models/Modal-AI-Training`, workspace
`danielrosehill`), same setup as ASR-Paragraph-Breaks.

```bash
uv run scripts/extrapolate.py label --model claude-sonnet-5-5   # data/silver.jsonl (gitignored)
uv run --with anthropic --with huggingface_hub scripts/build_training.py
export MODAL_PROFILE=danielrosehill
cd ~/repos/github/ai-models/Modal-AI-Training
modal run --detach app.py --upload ~/repos/github/audio-dictation/Email-Key-Line-Bolding/training \
  --cmd "EPOCHS=10 python train.py" --run email-bold-r2-gemma270m-e10 --gpu L4
```

Split (2026-10-21): train 336 = 13 gold + 323 silver (Sonnet 5.5 with `docs/bolding-style.md` and the
13 as examples); test = the 6 gold in `extrapolate.TEST_IDS`, never shown to the labeller. Exact copy:
dataset `danielrosehill/Email-Key-Line-Bolding`, `training/r2/{train,test}.jsonl` and `silver/silver.jsonl`.
Scoring is per word (see `training/train.py`); an output that changes anything but `**` is a fallback.

## Runs

Base `google/gemma-3-270m-it` (chosen in ASR-Paragraph-Breaks `docs/training.md`), L4, batch 4 × 2.

| Run | Epochs | LR | Train time | Word P / R / F1 | Fallback | Notes |
|---|---|---|---|---|---|---|
| email-bold-r1-gemma270m-e3 | 3 | 5e-5 | 158 s | 0.65 / 0.28 / 0.39 | 1 of 6 | under-bolds |
| email-bold-r1-gemma270m-e6 | 6 | 5e-5 | 312 s | 0.65 / 0.47 / 0.54 | 1 of 6 | |
| email-bold-r2-gemma270m-e10 | 10 | 5e-5 | 504 s | 0.73 / 0.54 / 0.62 | 0 of 6 | **Published (private)** |
| email-bold-r3-gemma270m-e6-lr1e4 | 6 | 1e-4 | 307 s | 0.74 / 0.44 / 0.56 | 0 of 6 | |

Untrained base: 6 of 6 fallback. Labeller reference: Sonnet 5.5 word F1 0.69 (4-fold CV over all 19
gold). F1 was still rising at 10 epochs; more epochs or more gold are the next things to try. Per-run
`metrics.json` and `pred_ft.jsonl` are in `runs/<run>/`.

## Where the weights are (2026-10-21)

| Copy | Location |
|---|---|
| Hub (private) | [`danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding`](https://huggingface.co/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding), r2-e10; card source `hub/model/README.md`; `model.safetensors` LFS sha256 `e080dfa26ec1…` matches the archive |
| Archive (private) | `wasabi:small-models/ai-training/runs/email-bold-r2-gemma270m-e10/` with `SHA256SUMS`; `rclone check` 0 differences, 12 files |
| Modal volume `ai-training` | `model/` and `ckpt/` deleted for all four runs after the checks; `metrics.json`, `pred_*.jsonl`, `job.log` kept. The r1/r3 weights were not archived (retrain from the command above) |

Closed out with the `custom-model-planning:finish-training-run` skill (added 2026-10-21 for this).
