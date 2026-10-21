# Bolding style

What Daniel bolds in a dictated email, derived from the first 19 hand annotations (2026-10-21,
`annotations/` in the HF dataset). The same rules, worded as a prompt, are `SYSTEM` in
[`scripts/extrapolate.py`](../scripts/extrapolate.py); keep the two in step.

**The test:** someone reading only the bold knows what the email is about and what is being asked of them.

## Bold

1. **The ask.** Every direct request or question to the recipient ("Could you let me know…", "Can you
   confirm how many people are coming?", "Reply to this email by Thursday…"). 15 of 87 spans contain a
   `?`, and nearly every email's request is bold.
2. **The purpose line**, when the email states why it is written ("I'm writing to ask a few questions
   before I decide whether to submit an application", "I wanted to let you know because…").
3. **Each point of a list.** The sentence carrying "First," / "Second," / "Third," / "Also," / "Finally,"
   is bold (syn-1-0163, syn-1-0485, syn-3-0170, syn-3-0172, syn-3-0205).
4. **The key facts to act on**: the decisive date, time, place, amount, model number or the problem
   itself ("still hasn't arrived", "The stand's hydraulic arm is leaking.", "$287.40… almost double").
5. **Instructions and warnings** ("Don't use it for anything heavy until Marcus has looked at it.",
   "Please don't proceed with any work above two hundred and fifty pounds without contacting me first").

## Don't bold

- **The greeting / opening paragraph.** In none of the 19 emails is anything in the first paragraph bold.
- Sign-offs, thanks, pleasantries.
- Background, reasons, apologies and asides that support a point rather than make it.
- Routine contact details, unless they are inside the ask (syn-3-0078, syn-3-0027 include them).

## Form

- A span is usually one whole sentence (median 16 words, range 3–63). A clause is bolded instead when only
  part of the sentence is the point: a lead-in "so" or a trailing "because…" is dropped (syn-3-0052,
  syn-3-0081 "The parcel with the printed storyboard sets," + "still hasn't arrived").
- Occasionally two consecutive sentences that make one request form one span (syn-3-0027, syn-3-0130).
- Density: median 25% of words bold (range 9–54%), 2–10 spans, median 1.2 spans per 100 words.
  Terse, list-like emails (syn-3-0205, syn-3-0129) get more; long narrative ones (syn-3-0143, syn-3-0258)
  get as little as 10%.

## How well an LLM reproduces it

`uv run scripts/extrapolate.py eval --model <m>`: 4-fold cross-validation over the 19 gold; the model
gets the rules above plus the other folds' annotations as examples. Per-word scoring. 2026-10-21:

| Labeller | Word P | Word R | Word F1 | Rejected (altered text) |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.65 | 0.62 | 0.63 | 1 of 19 |
| **claude-sonnet-5-5** | 0.65 | 0.73 | **0.69** | 0 |
| claude-opus-5-5 | 0.63 | 0.77 | 0.69 | 0 |

Sonnet was chosen to label the rest of the pool (silver): same F1 as Opus, bolds less. Both larger
models over-bold slightly (Sonnet 1783 bold words against 1571 gold). Worst agreement is on long
narrative emails where Daniel bolded very little (syn-3-0143 F1 0.16, syn-3-0136 0.21): the models
bold the ask correctly but add the background facts. There is no inter-annotator baseline (one
annotator), so whether 0.69 is near the ceiling for this subjective task is unknown. Re-run the eval
once there is more gold.
