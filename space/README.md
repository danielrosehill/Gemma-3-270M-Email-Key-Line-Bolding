---
title: Email Key-Line Annotator
emoji: 📧
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
short_description: Mark the key words to bold in dictated emails
---

# Email Key-Line Annotator

Private annotation Space for the (public since 2026-10-21) dataset
[`danielrosehill/Email-Key-Line-Bolding`](https://huggingface.co/datasets/danielrosehill/Email-Key-Line-Bolding).
Shows one dictated email at a time. Drag across any words to bold them (snaps to whole words, one
paragraph at most); click a bold span to remove it; or record "no bold". Then save. Each save is committed as `annotations/<id>.jsonl` with the signed-in user's own OAuth token,
which is also used to read the pool. No token is stored in the Space.

Source: [github.com/danielrosehill/Email-Key-Line-Bolding](https://github.com/danielrosehill/Email-Key-Line-Bolding) (`space/`, private).
