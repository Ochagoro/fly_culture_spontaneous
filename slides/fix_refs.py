"""Renumber appendix cross-references and rebuild the appendix table of contents in slides.html.

Run after adding, removing or reordering slides: python3 slides/fix_refs.py
- <a class="xref" href="#a-id">付録 N</a>  -> N = position of that appendix slide (付録 1 = first appendix slide)
- <a class="xref" href="#a-id">N</a>      -> bare number form (used in lists such as 付録 12〜14)
- <ol class="toc" id="toc"> is filled from data-toc attributes of the appendix slides.
"""
import re
from pathlib import Path

P = Path(__file__).resolve().parent / "slides.html"
s = P.read_text()

appx = re.findall(r'<section class="slide appx" id="([^"]+)"(?: data-toc="([^"]*)")?', s)
num = {sid: k + 1 for k, (sid, _) in enumerate(appx)}


def repl(m):
    sid, text = m.group(1), m.group(2)
    if sid not in num:
        raise SystemExit(f"unknown xref target {sid}")
    n = num[sid]
    new = f"付録 {n}" if text.startswith("付録") else str(n)
    return f'<a class="xref" href="#{sid}">{new}</a>'


s = re.sub(r'<a class="xref" href="#([^"]+)">([^<]*)</a>', repl, s)

items = [f'<li><span class="k">付録 {num[sid]}</span>{title}</li>' for sid, title in appx if title]
s = re.sub(r'(<ol class="toc" id="toc">).*?(</ol>)', lambda m: m.group(1) + "\n    " + "\n    ".join(items) + "\n  " + m.group(2),
           s, flags=re.S)
P.write_text(s)
print(f"{len(appx)} appendix slides; {len(items)} TOC entries")
