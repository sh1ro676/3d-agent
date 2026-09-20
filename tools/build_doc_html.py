#!/usr/bin/env python3
"""
Build the readable HTML version of a Markdown document.

Usage:
    python build_doc_html.py <input.md> [output.html]

Requires the `markdown` package (available in the Anaconda python at
D:\\Users\\ROG\\anaconda3\\python.exe).

The styling intentionally matches the light IDE theme: white content card on a
warm off-white page, single accent colour, no gradients or shadows.
"""

from __future__ import annotations

import html
import re
import sys
import pathlib

try:
    import markdown
except ImportError:
    sys.exit("ERROR: the 'markdown' package is missing. "
             "Run this with D:\\Users\\ROG\\anaconda3\\python.exe")

CSS = """
:root{--fg:#1a1a1a;--fg2:#5f5e5a;--bg:#ffffff;--bg2:#f7f6f3;--bd:#e3e1db;--acc:#185fa5;--accbg:#e6f1fb;--warn:#854f0b;--warnbg:#faeeda;--ok:#0f6e56;--okbg:#e1f5ee;--no:#a32d2d;--nobg:#fcebeb;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg2);color:var(--fg);font-family:-apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 80px;display:grid;grid-template-columns:250px minmax(0,1fr);gap:36px;align-items:start}
nav{position:sticky;top:24px;max-height:calc(100vh - 48px);overflow:auto;padding:20px 0;font-size:12.5px;line-height:1.5}
nav .nvt{font-weight:500;font-size:12px;color:var(--fg2);letter-spacing:.06em;margin-bottom:10px;text-transform:uppercase}
nav a{display:block;color:var(--fg2);text-decoration:none;padding:3px 0 3px 9px;border-left:2px solid transparent}
nav a:hover{color:var(--acc);border-left-color:var(--acc)}
nav a.t2{padding-left:9px}
nav a.t3{padding-left:22px;font-size:12px;color:#88867f}
main{background:var(--bg);border:1px solid var(--bd);border-radius:12px;padding:44px 52px;min-width:0}
h1{font-size:27px;font-weight:500;line-height:1.35;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:21px;font-weight:500;margin:48px 0 14px;padding-top:22px;border-top:1px solid var(--bd);letter-spacing:-.01em}
h2:first-of-type{border-top:none;padding-top:0}
h3{font-size:17px;font-weight:500;margin:32px 0 10px;color:#0f0f0f}
h4{font-size:15px;font-weight:500;margin:24px 0 8px;color:var(--fg2)}
p{margin:12px 0}
strong{font-weight:500;color:#0f0f0f}
code{font-family:"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;font-size:.875em;background:var(--bg2);border:1px solid var(--bd);border-radius:4px;padding:1px 5px;color:#7a3d0a;word-break:break-word}
pre.code{background:#fbfaf8;border:1px solid var(--bd);border-radius:9px;padding:15px 18px;overflow-x:auto;margin:16px 0;line-height:1.6}
pre.code code{background:none;border:none;padding:0;color:#26251f;font-size:12.6px;white-space:pre}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:13.4px;display:block;overflow-x:auto}
th,td{border:1px solid var(--bd);padding:8px 11px;text-align:left;vertical-align:top}
th{background:var(--bg2);font-weight:500;white-space:nowrap}
tbody tr:nth-child(even){background:#fcfbf9}
ul,ol{margin:12px 0;padding-left:24px}
li{margin:5px 0}
hr{border:none;border-top:1px solid var(--bd);margin:36px 0}
blockquote{margin:18px 0;padding:13px 18px;background:var(--accbg);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;color:#12456f}
blockquote p{margin:0}
h2+p,h3+p{margin-top:8px}
@media(max-width:940px){.wrap{grid-template-columns:1fr;gap:0}nav{position:static;max-height:none;border-bottom:1px solid var(--bd);margin-bottom:20px}main{padding:28px 22px;border-radius:10px}}
"""

HEAD_RE = re.compile(r'<(h[123])\s+id="([^"]+)"[^>]*>(.*?)</\1>', re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def build_toc(body: str) -> str:
    parts = ['<nav><div class="nvt">目录</div>']
    for m in HEAD_RE.finditer(body):
        level, anchor, inner = m.group(1).lower(), m.group(2), m.group(3)
        label = html.unescape(TAG_RE.sub("", inner)).strip()
        cls = {"h1": "t1", "h2": "t2", "h3": "t3"}[level]
        parts.append(f'<a class="{cls}" href="#{anchor}">{html.escape(label)}</a>')
    parts.append("</nav>")
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    text = src.read_text(encoding="utf-8")

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={"toc": {"permalink": False}},
    )

    body = body.replace("<pre><code", '<pre class="code"><code')
    body = body.replace('<pre class="code"><code class="language-', '<pre class="code"><code class="language-')

    title = src.stem
    first_h1 = HEAD_RE.search(body)
    if first_h1:
        title = html.unescape(TAG_RE.sub("", first_h1.group(3))).strip() or title

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{build_toc(body)}
<main>
{body}
</main>
</div>
</body>
</html>
"""
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        f.write(doc)

    print(f"wrote {dst}")
    print(f"  source : {src}  ({len(text)} chars)")
    print(f"  output : {len(doc)} chars")
    print(f"  headings in toc : {len(HEAD_RE.findall(body))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
