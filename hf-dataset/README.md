---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
tags:
  - email
  - dictation
  - emphasis
  - text-formatting
pretty_name: Email Key-Line Bolding
size_categories:
  - n<1K
configs:
  - config_name: annotations
    default: true
    data_files: annotations/*.jsonl
  - config_name: pool
    data_files: pool/pool.jsonl
---

# Email Key-Line Bolding

Hand annotations of which words in a dictated email should be **bold**: any run of whole words within
a paragraph, from a phrase to several sentences. The task for the model trained on it: given an email's
text, return the same text with its key spans wrapped in `**…**`
(Markdown bold), or unchanged when nothing deserves bold. No other change is allowed, so input and
output differ only by the `**` markers.

**Public** since 2026-10-21. All emails are synthetic (Claude Haiku 5.5); the annotation Space is private.

## Configs

| Config | File(s) | What |
|---|---|---|
| `annotations` (default) | `annotations/<id>.jsonl` | One hand annotation per email |
| `pool` | `pool/pool.jsonl` | Every email waiting for annotation, split into paragraphs and sentences |
| — | `silver/silver.jsonl` | LLM labels (Claude Sonnet 5.5, from the written rules and the gold as examples) for the emails not yet hand-annotated. Fields: `id`, `kind`, `text`, `bold_spans`, `bolded`, `labeller`. Not gold |
| — | `training/r2/{train,test}.jsonl` | Exact split used for the preliminary model `danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding`: `{id, label: gold\|silver, input, output}` |

Training pair: `text` → `bolded`. Field reference: `docs/data-format.md` in the GitHub repo.

## Source

342 synthetic dictated emails to 12 kinds of recipient (colleague, landlord, client, supplier, school,
doctor's office, HR, contractor, bank/insurer/utility, volunteers, freelancer, venue), written by Claude
Haiku 5.5 for [`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks),
with that project's paragraphing: hand-reviewed where available, otherwise its rule-based proposal.
Unit of emphasis is any run of whole words inside one paragraph (spans are stored as character offsets).

Annotate in the private Space `danielrosehill/Email-Key-Line-Annotator`.
Build scripts: github.com/danielrosehill/Email-Key-Line-Bolding (private).
