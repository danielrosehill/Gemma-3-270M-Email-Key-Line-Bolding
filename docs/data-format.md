# Data format

Dataset repo (private): `danielrosehill/Email-Key-Line-Bolding`.

## `pool/pool.jsonl`

Built by `scripts/build_pool.py`. One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `id`, `source_id` | string | The item's id in ASR-Paragraph-Breaks (`syn-<seed>-<n>`), kept unchanged |
| `kind`, `topic`, `style` | string | Generation spec from ASR-Paragraph-Breaks |
| `paragraph_source` | string | `gold` (hand-reviewed paragraphing) or `style_rules v1.1` |
| `paragraphs` | list[string] | The email body, one string per paragraph |
| `sentences` | list[list[string]] | Each paragraph split into sentences (`style_rules.sentences`) |
| `words` | int | Word count |
| `generator` | string | Model that wrote the email |

## `annotations/<id>.jsonl`

Written by the Space (`space/app.py`), one single-line file per email. In local mode the same file is
written to `data/annotations/<id>.jsonl` first and pushed to the dataset in a batched commit; the HF
copy is the canonical one.

| Field | Type | Meaning |
|---|---|---|
| `id`, `source_id`, `kind` | string | Copied from the pool item |
| `paragraphs`, `sentences` | list | Copied from the pool item |
| `text` | string | Input: paragraphs joined by `\n\n` |
| `bold_spans` | list[[int, int]] | `[start, end)` character offsets into `text`, sorted, non-overlapping. Each span starts and ends on a word boundary and contains no `\n\n` (validated on save) |
| `bold_texts` | list[string] | `text[start:end]` for each span |
| `none` | bool | True when no span was saved (the explicit "No bold (save)" button) |
| `bolded` | string | Target output: `text` with each span wrapped in `**…**` |
| `n_bold` | int | `len(bold_spans)` |
| `annotator`, `annotated_at` | string | HF username; UTC ISO 8601 |

Invariant: removing every `**` from `bolded` gives `text` exactly.

Offsets are Python code-point indices. The browser computes them in UTF-16 code units, which agree for
all characters in the Basic Multilingual Plane (letters, `£`, `€`, accented text); an emoji or other
astral character before a span would shift it. The synthetic emails contain none; if real text is ever
added, check for that.

History: `sentences` stays in the record as metadata. The first UI stored whole sentences as
`bold: [[paragraph, sentence], ...]`. One annotation was saved that way (`syn-3-0004`, 2026-10-20) and
was converted to `bold_spans` on 2026-10-21; it carries `migrated_from`.

## `silver/silver.jsonl` and `training/r2/`

Written by `scripts/extrapolate.py label` and `scripts/build_training.py` (2026-10-21). Silver rows are
`{id, kind, text, bold_spans, bolded, labeller}`, spans normalised the same way the annotator does
(whole words, one paragraph, merged). Training rows are `{id, label, input, output}` with `output` =
`bolded`. Gold always wins: an id with a hand annotation is never taken from silver.
