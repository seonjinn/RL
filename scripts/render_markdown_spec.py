#!/usr/bin/env python3
"""Render a Markdown experiment specification as a standalone Pages HTML file."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import markdown


def page_template(title: str, body: str) -> str:
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root{{--text:#111827;--muted:#6b7280;--line:#d8dee8;--bg:#f7f8fb;--panel:#fff;--blue:#1f5fbf}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.5}}
    main{{max-width:1120px;margin:0 auto;padding:24px}}
    .topbar{{margin-bottom:18px}}
    .topbar a{{display:inline-flex;border:1px solid var(--line);border-radius:8px;background:#fff;padding:6px 10px;color:var(--blue);font-weight:700;text-decoration:none}}
    article{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:24px}}
    h1{{font-size:28px;margin:0 0 18px}}h2{{font-size:20px;margin:28px 0 10px}}h3{{font-size:16px;margin:20px 0 8px}}
    p,li{{max-width:90ch}}code{{background:#f3f4f6;border-radius:4px;padding:1px 4px}}
    pre{{overflow:auto;background:#111827;color:#f9fafb;border-radius:8px;padding:14px}}pre code{{background:transparent;padding:0}}
    table{{width:100%;border-collapse:collapse;margin:12px 0 20px}}th,td{{border:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}th{{background:#eef2f7}}
    @media(max-width:720px){{main{{padding:12px}}article{{padding:16px;overflow-x:auto}}table{{min-width:720px;font-size:13px}}}}
  </style>
</head>
<body><main>
  <div class="topbar"><a href="../index.html">Back to report hub</a></div>
  <article>{body}</article>
</main></body>
</html>
"""


def render_spec(markdown_path: Path, output_path: Path, title: str) -> None:
    body = markdown.markdown(
        markdown_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page_template(title, body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_spec(args.markdown_path, args.output_path, args.title)
    print(args.output_path)


if __name__ == "__main__":
    main()
