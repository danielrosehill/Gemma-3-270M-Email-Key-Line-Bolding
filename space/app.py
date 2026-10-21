"""Key-line bolding annotator for dictated emails.

Shows one email at a time. The annotator drag-selects any run of words to bold it (the selection snaps
outward to whole words and is clipped to one paragraph); clicking a bold span removes it; overlapping
or touching spans merge. "No bold (save)" records that an email needs none. Only the spans are
recorded; the text is never edited.

Spans are character offsets [start, end) into the canonical email text: paragraphs joined by "\\n\\n".
The email is gr.HTML with one <p data-off data-text> per paragraph; a script in Blocks(head=...) handles
selection and re-renders client-side at once (gr.HTML does not run inline scripts). It mirrors the
span list as JSON into a CSS-hidden Textbox (#sel-box), which the save handlers read and re-validate.
A visible=False component would not be in the DOM, hence the CSS hiding.

Each save is committed to the private dataset repo as annotations/<id>.jsonl (one line) with the
signed-in user's own OAuth token, so no token is stored in the Space and only accounts with write
access to the dataset can save. Patterns follow danielrosehill/ASR-Paragraph-Breaks space/app.py.

Local mode (ANNOTATOR_LOCAL=1, see scripts/annotate_local.sh): no sign-in; the local HF token
(huggingface_hub.get_token()) is used instead. On the Space a save takes ~20 s, almost all of it the
synchronous HF commit plus a list_repo_files on every pick. Locally each save is written to
data/annotations/<id>.jsonl at once and a background thread commits pending files to the dataset in
batches, so the next email appears immediately. The done set is listed from HF once at startup and
then kept in memory. Any local file not yet on HF (e.g. the app was killed mid-upload) is re-queued
at the next start.
"""

from __future__ import annotations

import html
import json
import os
import queue
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

DATASET = os.environ.get("DATASET_REPO", "danielrosehill/Email-Key-Line-Bolding")
POOL_PATH = "pool/pool.jsonl"
# The dataset is private: reading it needs a token. The Space's own token (HF_TOKEN secret) is not
# set, so reads use the signed-in user's OAuth token, passed into every handler.

LOCAL = os.environ.get("ANNOTATOR_LOCAL") == "1"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "data" / "annotations"

WORD = re.compile(r"[\w'’]")  # must match isWord() in HEAD


def load_pool(token: str) -> list[dict]:
    path = hf_hub_download(DATASET, POOL_PATH, repo_type="dataset", token=token, force_download=True)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def remote_done_ids(token: str) -> set[str]:
    files = HfApi(token=token).list_repo_files(DATASET, repo_type="dataset")
    return {f[len("annotations/"):-len(".jsonl")] for f in files if f.startswith("annotations/") and f.endswith(".jsonl")}


DONE: set[str] = set()  # local mode only: the done set, kept in memory after one listing
PENDING: queue.Queue[str] = queue.Queue()  # local mode only: ids saved locally, not yet on HF


def done_ids(token: str) -> set[str]:
    return DONE if LOCAL else remote_done_ids(token)


def uploader(token: str) -> None:
    """Commit locally saved annotations to the dataset, batching whatever is pending into one commit."""
    api = HfApi(token=token)
    while True:
        batch = {PENDING.get()}
        time.sleep(2)  # let a few quick saves pile up into the same commit
        while not PENDING.empty():
            batch.add(PENDING.get())
        ops = [CommitOperationAdd(f"annotations/{i}.jsonl", str(LOCAL_DIR / f"{i}.jsonl")) for i in sorted(batch)]
        try:
            api.create_commit(DATASET, ops, repo_type="dataset",
                              commit_message=f"Annotate {len(ops)} email(s): {', '.join(sorted(batch))}")
            print(f"pushed {len(ops)}: {', '.join(sorted(batch))}", flush=True)
        except Exception as e:  # network blip: keep the files queued and retry
            print(f"push failed ({e}); retrying in 30 s", flush=True)
            for i in batch:
                PENDING.put(i)
            time.sleep(30)


def start_local() -> None:
    token = get_token()
    if not token:
        raise SystemExit("No local HF token: run `hf auth login` or set HF_TOKEN.")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    remote = remote_done_ids(token)
    local = {p.stem for p in LOCAL_DIR.glob("*.jsonl")}
    DONE.update(remote | local)
    for i in sorted(local - remote):
        PENDING.put(i)
    threading.Thread(target=uploader, args=(token,), daemon=True).start()
    print(f"local mode: {len(remote)} on HF, {len(local - remote)} local-only re-queued", flush=True)


POOL: dict[str, dict] = {}  # filled on first sign-in


def ensure_pool(token: str) -> None:
    if not POOL:
        POOL.update({r["id"]: r for r in load_pool(token)})


def text_of(item: dict) -> str:
    return "\n\n".join(item["paragraphs"])


def email_html(item: dict) -> str:
    paras, off = [], 0
    for p in item["paragraphs"]:
        paras.append(f'<p data-off="{off}" data-text="{html.escape(p, quote=True)}">{html.escape(p)}</p>')
        off += len(p) + 2
    return f'<div id="email">{"".join(paras)}</div>'


def is_word(ch: str) -> bool:
    return bool(ch) and bool(WORD.match(ch))


def parse_spans(item: dict, raw: str) -> list[list[int]]:
    """JSON [[a, b], ...] from the hidden textbox -> validated, sorted, merged spans."""
    text = text_of(item)
    try:
        spans = sorted([int(a), int(b)] for a, b in json.loads(raw or "[]"))
    except (json.JSONDecodeError, TypeError, ValueError):
        raise gr.Error("Selection could not be read; press Clear and try again.")
    merged: list[list[int]] = []
    for a, b in spans:
        ok = (0 <= a < b <= len(text) and "\n\n" not in text[a:b]
              and not (a > 0 and is_word(text[a - 1]) and is_word(text[a]))
              and not (b < len(text) and is_word(text[b - 1]) and is_word(text[b])))
        if not ok:
            raise gr.Error("A bold span is out of range or splits a word; press Clear and redo it.")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def bolded(text: str, spans: list[list[int]]) -> str:
    out, pos = [], 0
    for a, b in spans:
        out += [text[pos:a], "**", text[a:b], "**"]
        pos = b
    return "".join(out + [text[pos:]])


def pick_next(token: str, skipped: list[str]):
    ensure_pool(token)
    done = done_ids(token)
    todo = [r for r in POOL.values() if r["id"] not in done and r["id"] not in skipped]
    progress = f"{len(done)} annotated · {len(todo)} left in this session's queue · {len(POOL)} in pool"
    # Shortest first (Daniel, 2026-10-21: the long emails are tedious); random among the 10 shortest left.
    todo.sort(key=lambda r: r["words"])
    return (random.choice(todo[:10]) if todo else None), progress


def show(item: dict | None, progress: str):
    if item is None:
        return None, "", "[]", "Nothing left to annotate.", progress
    meta = f"**{item['id']}** · {item['kind']} · {item['topic']} · {item['words']} words"
    return item["id"], email_html(item), "[]", meta, progress


def need_token(oauth_token: gr.OAuthToken | None) -> str:
    if LOCAL:
        return get_token()
    if oauth_token is None:
        raise gr.Error("Sign in with Hugging Face first.")
    return oauth_token.token


LOCAL_USER: list[str] = []  # local mode: HF username, looked up once


def username(profile: gr.OAuthProfile | None) -> str | None:
    if not LOCAL:
        return profile.username if profile else None
    if not LOCAL_USER:
        LOCAL_USER.append(HfApi(token=get_token()).whoami()["name"])
    return LOCAL_USER[0]


def on_load(skipped, oauth_token: gr.OAuthToken | None):
    if oauth_token is None and not LOCAL:
        return None, "", "[]", "Sign in to start.", ""
    return show(*pick_next(need_token(oauth_token), skipped or []))


def on_clear(item_id):
    return (email_html(POOL[item_id]) if item_id else ""), "[]"


def on_skip(item_id, skipped, oauth_token: gr.OAuthToken | None):
    token = need_token(oauth_token)
    skipped = (skipped or []) + ([item_id] if item_id else [])
    return (*show(*pick_next(token, skipped)), skipped)


def save(item_id, raw, skipped, profile, oauth_token, none: bool):
    token = need_token(oauth_token)
    user = username(profile)
    if user is None or not item_id:
        raise gr.Error("Sign in and load an item first.")
    item = POOL[item_id]
    text = text_of(item)
    spans = [] if none else parse_spans(item, raw)
    if not none and not spans:
        raise gr.Error("Nothing bolded. Use 'No bold (save)' to record that this email needs no bold.")
    record = {
        "id": item_id,
        "source_id": item["source_id"],
        "kind": item["kind"],
        "paragraphs": item["paragraphs"],
        "sentences": item["sentences"],
        "text": text,
        "bold_spans": spans,
        "bold_texts": [text[a:b] for a, b in spans],
        "none": not spans,
        "bolded": bolded(text, spans),
        "n_bold": len(spans),
        "annotator": user,
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    if LOCAL:
        (LOCAL_DIR / f"{item_id}.jsonl").write_bytes(line)
        DONE.add(item_id)
        PENDING.put(item_id)
    else:
        HfApi(token=token).upload_file(
            path_or_fileobj=line,
            path_in_repo=f"annotations/{item_id}.jsonl",
            repo_id=DATASET,
            repo_type="dataset",
            commit_message=f"Annotate {item_id} ({len(spans)} bold)",
        )
    gr.Info(f"Saved {item_id}: {len(spans)} bold span(s).")
    return (*show(*pick_next(token, skipped or [])), skipped)


def on_save(item_id, raw, skipped, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    return save(item_id, raw, skipped, profile, oauth_token, none=False)


def on_none(item_id, raw, skipped, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    return save(item_id, raw, skipped, profile, oauth_token, none=True)


CSS = """
#email p { margin: 0 0 1em; line-height: 1.7; cursor: text; }
#email b.bs { font-weight: 700; background: rgba(255, 200, 0, 0.30); border-radius: 3px; cursor: pointer; }
#email b.bs:hover { text-decoration: line-through; }
.hidden-box { display: none !important; }
"""

# Document-level listeners, so they keep working after Gradio swaps in the next email's HTML.
# Spans live in the hidden textbox (#sel-box); the textbox is the single source of truth.
HEAD = r"""
<script>
(() => {
  const isWord = (c) => !!c && /[\w'’]/.test(c);
  const box = () => document.querySelector("#sel-box textarea");
  const getSpans = () => { try { return JSON.parse(box().value || "[]"); } catch { return []; } };
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function setSpans(spans) {
    spans.sort((x, y) => x[0] - y[0]);
    const merged = [];
    for (const [a, b] of spans) {
      if (merged.length && a <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], b);
      else merged.push([a, b]);
    }
    const el = box();
    if (el) { el.value = JSON.stringify(merged); el.dispatchEvent(new Event("input", { bubbles: true })); }
    render(merged);
  }

  function render(spans) {
    document.querySelectorAll("#email p").forEach((p) => {
      const off = +p.dataset.off, text = p.dataset.text;
      let out = "", pos = 0;
      for (const [a, b] of spans) {
        const la = a - off, lb = b - off;
        if (lb <= 0 || la >= text.length) continue;
        out += esc(text.slice(pos, la)) + `<b class="bs" data-a="${a}" data-b="${b}">` + esc(text.slice(la, lb)) + "</b>";
        pos = lb;
      }
      p.innerHTML = out + esc(text.slice(pos));
    });
  }

  function localOffset(p, node, offset) {
    const r = document.createRange();
    r.setStart(p, 0);
    r.setEnd(node, offset);
    return r.toString().length;
  }

  document.addEventListener("mouseup", (e) => {
    const sel = window.getSelection();
    const bs = e.target.closest && e.target.closest("#email b.bs");
    if (sel && sel.isCollapsed && bs) {  // a plain click on a bold span removes it
      setSpans(getSpans().filter(([a, b]) => !(a === +bs.dataset.a && b === +bs.dataset.b)));
      return;
    }
    if (!sel || sel.isCollapsed || !sel.rangeCount) return;
    const range = sel.getRangeAt(0);
    const startNode = range.startContainer.nodeType === 3 ? range.startContainer.parentNode : range.startContainer;
    const p = startNode.closest && startNode.closest("#email p");
    if (!p) return;
    const text = p.dataset.text;
    let la = localOffset(p, range.startContainer, range.startOffset);
    let lb = p.contains(range.endContainer) ? localOffset(p, range.endContainer, range.endOffset) : text.length;  // clip to one paragraph
    while (la < lb && /\s/.test(text[la])) la++;
    while (lb > la && /\s/.test(text[lb - 1])) lb--;
    while (la > 0 && isWord(text[la - 1]) && isWord(text[la])) la--;
    while (lb < text.length && isWord(text[lb - 1]) && isWord(text[lb])) lb++;
    sel.removeAllRanges();
    if (lb <= la) return;
    const off = +p.dataset.off;
    setSpans([...getSpans(), [off + la, off + lb]]);
  });
})();
</script>
"""

with gr.Blocks(title="Email Key-Line Annotator", css=CSS, head=HEAD) as demo:
    gr.Markdown(
        "# Email key-line annotator\n"
        "**Drag across any words to bold them** (the selection snaps to whole words); **click a bold span to remove it**. "
        "Then **Save & next**. If the email needs no bold at all, press **No bold (save)**. "
        "The text itself is never edited. Sign in first (the dataset is private)."
    )
    if not LOCAL:
        gr.LoginButton()
    item_id = gr.State(None)
    skipped = gr.State([])
    meta = gr.Markdown()
    progress = gr.Markdown()
    email = gr.HTML()
    picks = gr.Textbox(value="[]", elem_id="sel-box", elem_classes=["hidden-box"], show_label=False)
    with gr.Row():
        save_btn = gr.Button("Save & next", variant="primary")
        none_btn = gr.Button("No bold (save)")
        clear_btn = gr.Button("Clear")
        skip_btn = gr.Button("Skip")

    outs = [item_id, email, picks, meta, progress]
    demo.load(on_load, [skipped], outs)
    save_btn.click(on_save, [item_id, picks, skipped], [*outs, skipped])
    none_btn.click(on_none, [item_id, picks, skipped], [*outs, skipped])
    clear_btn.click(on_clear, [item_id], [email, picks])
    skip_btn.click(on_skip, [item_id, skipped], [*outs, skipped])

if __name__ == "__main__":
    if LOCAL:
        start_local()
    demo.launch()
