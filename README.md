# Email Key-Line Bolding

A small text-in/text-out model that puts the key words of a dictated email in **bold** (any run of
whole words: a phrase, a sentence, or more), or leaves the email alone when nothing needs it, and
changes nothing else.

**Annotation UI (2026-10-21):** drag across words to bold them; the selection snaps outward to whole
words and is clipped to one paragraph. Click a bold span to remove it; overlapping or touching spans
merge. The first version (ticking whole sentences from a list) was replaced at Daniel's request,
first by double-click-a-sentence and then, before shipping, by free word spans, because he sometimes
wants to bold part of a sentence.

Sibling of [ASR-Paragraph-Breaks](https://github.com/danielrosehill/ASR-Paragraph-Breaks): the emails
are the synthetic dictated emails from that project, already paragraphed. This step comes after
paragraphing in the dictation pipeline.

**Status (2026-10-21): preliminary model.** 19 emails hand-annotated; the pattern is written up in
[`docs/bolding-style.md`](docs/bolding-style.md) and Sonnet 5.5 extrapolated it to the other 323 (word F1
0.69 against the gold in cross-validation). A Gemma 3 270M fine-tune on those scores word F1 0.62 on 6
held-out gold emails ([`docs/training.md`](docs/training.md)). Annotation continues, shortest emails first.
**Visibility (2026-10-21):** the model, the dataset and the demo Space are public. This GitHub repo and
the annotation Space stay private; the public code is
[`danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding`](https://github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding),
a copy of the tracked files here with fresh git history (so nothing from the private history is
published). Refresh it with `scripts/publish_public.sh`.

| | |
|---|---|
| Dataset (public) | [`danielrosehill/Email-Key-Line-Bolding`](https://huggingface.co/datasets/danielrosehill/Email-Key-Line-Bolding) |
| Annotation Space (private) | `danielrosehill/Email-Key-Line-Annotator` |
| Demo Space (public) | [`danielrosehill/Email-Key-Line-Bolding-Demo`](https://huggingface.co/spaces/danielrosehill/Email-Key-Line-Bolding-Demo), source [`demo/`](demo/) |
| Public code | [`danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding`](https://github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding) |
| Site page | <https://danielrosehill.com/ai/models/gemma-3-270m-email-key-line-bolding> |
| Model (public) | [`danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding`](https://huggingface.co/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding) (preliminary) |

## Layout

| Path | What |
|---|---|
| [`scripts/build_pool.py`](scripts/build_pool.py) | Selects the emails from the sibling repo's synthetic data, splits them into sentences, writes `pool/pool.jsonl`; `--upload` pushes it. Selection rules are in its docstring |
| [`scripts/style_rules.py`](scripts/style_rules.py) | Copy of the sibling repo's paragraphing rules and sentence splitter (origin commit in its docstring) |
| [`demo/`](demo/) | The public demo Space (CPU; optional paragraph-breaks step; bold or highlight view), deployed as-is |
| [`space/`](space/) | The Gradio annotation Space, deployed to HF as-is; `ANNOTATOR_LOCAL=1` runs it locally |
| [`scripts/annotate_local.sh`](scripts/annotate_local.sh) | Runs the annotator locally with background push (see below) |
| [`hf-dataset/README.md`](hf-dataset/README.md) | Dataset card |
| [`docs/data-format.md`](docs/data-format.md) | Pool and annotation schemas |
| [`docs/bolding-style.md`](docs/bolding-style.md) | The bolding pattern derived from the hand annotations, and how well LLMs reproduce it |
| [`scripts/extrapolate.py`](scripts/extrapolate.py) | Cross-validates an LLM-with-rules against gold (`eval`) and labels the rest of the pool (`label`) |
| [`scripts/build_training.py`](scripts/build_training.py), [`training/`](training/) | Train/test split and the Modal fine-tune script |
| [`docs/training.md`](docs/training.md) | Runs, results, where the weights are |
| [`hub/model/README.md`](hub/model/README.md) | Model card |
| [`project.json`](project.json) | Machine-readable project state |

## Annotating locally (faster)

```bash
scripts/annotate_local.sh     # then open http://127.0.0.1:7860
```

Same UI, no sign-in (uses the local HF token). On the Space each save takes ~20 s because it is a
synchronous HF commit plus a full `list_repo_files` before the next email. Locally a save writes
`data/annotations/<id>.jsonl` (gitignored) and returns in ~0.2 s; a background thread commits pending
files to the dataset in batches, and the next email loads in ~0.1 s. Leave it running a few seconds
after the last save; anything not yet pushed is re-queued on the next start. Measured 2026-10-21.

## Deploying

```bash
export HF_TOKEN="$HUGGINGFACE_TOKEN"
python3 scripts/build_pool.py --upload        # needs ../ASR-Paragraph-Breaks/data/synthetic (gitignored there)
hf upload danielrosehill/Email-Key-Line-Bolding hf-dataset/README.md README.md --repo-type dataset
hf upload danielrosehill/Email-Key-Line-Annotator space . --repo-type space
hf upload danielrosehill/Email-Key-Line-Bolding-Demo demo . --repo-type space   # public demo
scripts/publish_public.sh                                                          # public GitHub copy
```

The Space has no secrets. It reads the private pool and saves annotations with the signed-in user's
OAuth token (`hf_oauth_scopes: read-repos, write-repos`), so it only works for accounts with access
to the dataset.

## More emails

Seed 1 gave only 43 emails, so seed 3 (2026-10-21) generated 299 more, emails only, across 12 recipient
kinds: `uv run scripts/generate_synthetic.py run --n 300 --seed 3 --kinds email` in the sibling repo.
Rebuild the pool here afterwards; it picks up any `synthetic-<seed>.jsonl` whose kind starts
"a dictated email to". Target: about 200 annotated emails for a first fine-tune (bolding is sparse and
subjective, so it needs more examples than the paragraph task's proposal-review set).
