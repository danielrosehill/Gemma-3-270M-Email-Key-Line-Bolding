"""Public demo: bold (or highlight) the key lines of an email with Gemma-3-270M-Email-Key-Line-Bolding.

Optionally runs Gemma-3-270M-ASR-Paragraph-Breaks first, for flat dictated text. Both models run on the
Space's CPU, greedy, with the system prompts they were trained on (hub/model/README.md in each repo).
Each output is validated the way it would be in deployment: the bolding model may only add `**`, the
paragraph model may only change whitespace; otherwise that step falls back to its input.
"""
import html
import json
import re
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BOLD_REPO = "danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding"
PARA_REPO = "danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks"
BOLD_SYSTEM = ("Put the key parts of this email in Markdown bold (**like this**): the requests, the purpose, "
               "the main points and the key facts. Do not change, add or remove anything else.")
PARA_SYSTEM = ("Add paragraph breaks to this voice-note transcript. Separate paragraphs with a blank line. "
               "Do not change, add or remove any word or punctuation mark.")
MAX_WORDS = 700  # training emails are ~340 words median; long inputs are slow on CPU

torch.set_num_threads(max(1, torch.get_num_threads()))
MODELS = {}


def load(repo):
    if repo not in MODELS:
        MODELS[repo] = (AutoTokenizer.from_pretrained(repo),
                        AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval())
    return MODELS[repo]


load(BOLD_REPO)


def stream(repo, system, text):
    """Yield the growing greedy output of `repo` for `text`."""
    tok, model = load(repo)
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    n_in = enc["input_ids"].shape[1]
    Thread(target=model.generate, kwargs=dict(**enc, streamer=streamer, do_sample=False,
                                               max_new_tokens=int(n_in * 1.3) + 64)).start()
    out = ""
    for piece in streamer:
        out += piece
        yield out


def render(marked, mode):
    """Markdown-bold text -> HTML, as <strong> (outbound) or <mark> (inbound)."""
    tag = "mark" if mode.startswith("Inbound") else "strong"
    paras = []
    for para in re.split(r"\n\s*\n", marked.strip()):
        parts = re.split(r"(\*\*)", para)
        out, inside = [], False
        for p in parts:
            if p == "**":
                out.append(f"</{tag}>" if inside else f"<{tag}>")
                inside = not inside
            else:
                out.append(html.escape(p).replace("\n", "<br>"))
        if inside:
            out.append(f"</{tag}>")
        paras.append(f"<p>{''.join(out)}</p>")
    return f'<div class="email">{"".join(paras)}</div>'


def run(text, add_paragraphs, mode):
    text = (text or "").strip()
    if not text:
        yield "", "<p class='note'>Paste an email first.</p>", ""
        return
    if len(text.split()) > MAX_WORDS:
        yield "", f"<p class='note'>Please keep it under {MAX_WORDS} words (this runs on a small CPU).</p>", ""
        return
    notes = []
    if add_paragraphs:
        pred = text
        for pred in stream(PARA_REPO, PARA_SYSTEM, text):
            yield pred, "<p class='note'>Adding paragraph breaks…</p>", ""
        pred = pred.strip()
        if re.sub(r"\s+", "", pred) == re.sub(r"\s+", "", text):
            text = pred
        else:
            notes.append("The paragraph model changed more than whitespace, so its output was discarded.")
    marked = text
    for marked in stream(BOLD_REPO, BOLD_SYSTEM, text):
        yield marked, render(marked, mode), "Bolding…"
    marked = marked.strip()
    if marked.replace("**", "") != text or marked.count("**") % 2:
        notes.append("The bolding model changed more than the ** markers, so the email is shown unbolded "
                     "(this is the fallback a real pipeline would use).")
        marked = text
    elif "**" not in marked:
        notes.append("The model chose not to bold anything.")
    n = len(re.findall(r"\S+", text))
    bold = sum(len(re.findall(r"\S+", s)) for s in re.findall(r"\*\*(.+?)\*\*", marked, flags=re.S))
    notes.append(f"{bold} of {n} words marked ({bold / n:.0%}).")
    yield marked, render(marked, mode), " ".join(notes)


EXAMPLES = json.loads((Path(__file__).parent / "examples.json").read_text())

CSS = """
.email { font-size: 15px; line-height: 1.55; max-width: 68ch; }
.email p { margin: 0 0 0.9em; }
.email mark { background: #fde68a; color: inherit; padding: 0 2px; border-radius: 2px; }
.note { opacity: 0.7; }
"""

INTRO = """
# Email key-line bolding

A 270M-parameter Gemma 3 fine-tune that puts the key parts of an email in **bold** (the requests, the
purpose, the main points, the key facts) and changes nothing else. Built for dictation pipelines, where
it runs after a paragraph-breaks model. Two other ways to use it:

- **Outbound:** an AI agent drafting an email bolds the details so the recipient finds them at a glance.
- **Inbound:** mark up the emails you receive. The model only marks spans; a mail client could show them
  as highlights instead of bold. Pick *Inbound* below to see that.

Preliminary model (2026-10-21): word F1 0.62 on six held-out emails, precise but tends to under-bold.
The examples are held-out synthetic emails the model never saw. Runs on CPU, so allow 20-60 seconds.

[Model](https://huggingface.co/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding) ·
[Dataset](https://huggingface.co/datasets/danielrosehill/Email-Key-Line-Bolding) ·
[Code](https://github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding) ·
[Paragraph-breaks model](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks)
"""

with gr.Blocks(title="Email Key-Line Bolding", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Email", lines=16, placeholder="Paste an email (paragraphed, or tick the box below)")
            para = gr.Checkbox(label="Add paragraph breaks first (for flat dictated text)", value=False)
            mode = gr.Radio(["Outbound: bold", "Inbound: highlight"], value="Outbound: bold", label="Show as")
            btn = gr.Button("Mark key lines", variant="primary")
            gr.Examples([[e["text"], e["flat"]] for e in EXAMPLES], inputs=[inp, para],
                        example_labels=[("Flat dictation: " if e["flat"] else "") + e["text"].split("\n")[0][:60]
                                        for e in EXAMPLES])
        with gr.Column():
            out_html = gr.HTML(label="Rendered")
            status = gr.Markdown()
            with gr.Accordion("Raw output (Markdown)", open=False):
                out_md = gr.Textbox(lines=12, show_copy_button=True, label=None)
    btn.click(run, [inp, para, mode], [out_md, out_html, status])
    mode.change(lambda m, mode: render(m, mode) if m else "", [out_md, mode], out_html)

demo.queue(default_concurrency_limit=1).launch()
