#!/usr/bin/env python3
"""report.md -> report.pdf (one A4 page): Markdown -> HTML (python-markdown) -> PDF (headless Chrome).

    python3 scripts/build_report_pdf.py [--chrome PATH]

Needs the `markdown` package and Google Chrome/Chromium. Prints the page count of the result; the exam
requires report.pdf to be at most one page.
"""
import argparse
import pathlib
import re
import subprocess
import sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSS = """
@page { size: A4; margin: 11mm 12mm; }
body { font-family: "PingFang SC", "Noto Sans CJK SC", "Helvetica Neue", Arial, sans-serif;
       font-size: 8.8pt; line-height: 1.33; color: #111; }
h1 { font-size: 13pt; margin: 0 0 3pt; }
h2 { font-size: 10.5pt; margin: 8pt 0 3pt; border-bottom: 0.5pt solid #999; }
p, li { margin: 2pt 0; }
ul, ol { margin: 1pt 0 1pt 14pt; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.9pt; margin: 4pt 0; }
th, td { border: 0.4pt solid #888; padding: 1.5pt 3pt; text-align: center; }
th:first-child, td:first-child { text-align: left; }
img { display: block; max-width: 100%; max-height: 40mm; margin: 5pt auto 4pt; }
.row { display: flex; align-items: center; gap: 8pt; margin: 6pt 0; }
.row .fail { flex: 0 0 44%; } .row .fail table { margin: 0; } .row .fail td { white-space: nowrap; }
.row .fig { flex: 1; } .row img { max-height: 48mm; margin: 0 auto; }
code { font-size: 7.6pt; }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    a = ap.parse_args()
    md = (ROOT / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables"])
    # Put the failure-rate table and the figure that follows it side by side (uses the page width).
    body = re.sub(r"(<table>(?:(?!<table>).)*?运行失败率.*?</table>)\s*(<p><img[^>]*></p>)",
                  r'<div class="row"><div class="fail">\1</div><div class="fig">\2</div></div>', body, count=1, flags=re.S)
    html = f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    out_html = ROOT / "report.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = ROOT / "report.pdf"
    subprocess.run([a.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out_html.unlink()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    print(f"{pdf}: {pages} page(s)")
    return 0 if pages == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
