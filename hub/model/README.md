---
license: gemma
base_model: google/gemma-3-270m-it
language:
  - en
pipeline_tag: text-generation
library_name: transformers
datasets:
  - danielrosehill/Email-Key-Line-Bolding
tags:
  - dictation
  - email
  - text-emphasis
  - on-device
---

# Gemma 3 270M: email key-line bolding (preliminary)

Puts the key parts of a dictated email in Markdown **bold**, and changes nothing else. The input is
an email that has already been paragraphed (the step after
[`Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks)
in a dictation pipeline); the output is the same text with `**` around the requests, the purpose line,
the listed points and the key facts, so a reader skimming only the bold knows what is being asked.

**Status: first, preliminary version (2026-10-21).** Trained mostly on LLM-extrapolated labels from
19 hand annotations. It is a starting point to be replaced as more emails are annotated.

**Try it:** [Email Key-Line Bolding demo](https://huggingface.co/spaces/danielrosehill/Email-Key-Line-Bolding-Demo)
(CPU; can add paragraph breaks first; shows the result as bold or as highlights).

## Uses

- **Dictation / ASR pipelines** (the main one): after speech recognition and paragraphing, bold the lines the recipient must not miss.
- **Outbound email from AI agents:** bold the requests and details in an agent-drafted email so recipients find them at a glance. Pairs with the paragraph-breaks model as one post-processing step.
- **Inbound email:** mark up received mail. The model only marks spans; a client can render them as highlights instead of bold.

## Results

Scored on 6 held-out hand-annotated emails, per word: a word counts as bold if it falls inside a span.
An output that changes anything other than adding `**` is a fallback (the input is returned unbolded).

| Run | Epochs | LR | Word P | Word R | Word F1 | Fallback |
|---|---|---|---|---|---|---|
| r1-e3 | 3 | 5e-5 | 0.65 | 0.28 | 0.39 | 1 of 6 |
| r1-e6 | 6 | 5e-5 | 0.65 | 0.47 | 0.54 | 1 of 6 |
| **r2-e10 (this model)** | 10 | 5e-5 | 0.73 | 0.54 | **0.62** | 0 of 6 |
| r3-e6-lr1e-4 | 6 | 1e-4 | 0.74 | 0.44 | 0.56 | 0 of 6 |

Untrained base model: 6 of 6 fallbacks. Reference point: Claude Sonnet 5.5, prompted with the written
rules and the other annotations as examples, scores word F1 0.69 in 4-fold cross-validation over all 19
hand annotations. The model is precise but under-bolds (recall 0.54): it tends to bold the ask and the
purpose line and miss some of the listed points. Six test emails is a very small test set; treat the
numbers as rough.

## Training data

336 emails: 13 hand-annotated, 323 labelled by Claude Sonnet 5.5 from the written rules plus the 13 as
worked examples (silver). The emails are synthetic dictated emails (Claude Haiku 5.5) to 12 kinds of
recipient: landlords, schools, clinics, contractors, clients, clubs and so on. Median length about 340
words; median 25% of words bold in both the gold and the silver labels.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

SYSTEM = ("Put the key parts of this email in Markdown bold (**like this**): the requests, the purpose, "
          "the main points and the key facts. Do not change, add or remove anything else.")
email = "Hi Sam,\n\nCould you send me the signed lease by Friday?\n\nThanks,\nDana"
prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": email}],
                                 tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=1600, do_sample=False)
bolded = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# Validate: only ** may be added. Otherwise keep the input.
if bolded.replace("**", "") != email or bolded.count("**") % 2:
    bolded = email
```

Use the system prompt above exactly; it is the one the model was trained with.

## Limitations

- Bolding is subjective and the labels reflect one person's preference.
- Mostly silver labels; the model learns the LLM labeller's habits as well as the annotator's.
- English, synthetic dictated emails only; not tested on real transcripts or non-email text.
- Offsets are character-based; text with emoji has not been tested.

Training run: `email-bold-r2-gemma270m-e10` (Modal L4, 504 s). Source and rules:
[`docs/bolding-style.md`](https://github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding/blob/main/docs/bolding-style.md)
in [github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding](https://github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding).
