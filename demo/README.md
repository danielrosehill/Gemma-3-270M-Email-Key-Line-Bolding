---
title: Email Key-Line Bolding
emoji: ✉️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
models:
  - danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding
  - danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks
datasets:
  - danielrosehill/Email-Key-Line-Bolding
short_description: Bold or highlight the key lines of an email (Gemma 270M)
---

# Email Key-Line Bolding demo

Runs [`danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding`](https://huggingface.co/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding)
on CPU. Paste an email, optionally add paragraph breaks first with
[`Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks),
and view the key lines as bold (outbound) or highlights (inbound).

Source: [`demo/`](https://github.com/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding/tree/main/demo).
