"""Render memo/MEMO.md to a two-page PDF with a headless browser.

Usage:  python memo/build_pdf.py   ->  memo/MEMO.pdf, prints the page count.
Needs Edge or Chrome installed; no Python PDF libraries.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MD, HTML, PDF = HERE / "MEMO.md", HERE / "MEMO.html", HERE / "MEMO.pdf"

CSS = """
@page { size: A4; margin: 13mm 14mm; }
body { font-family: Calibri, "Segoe UI", Arial, sans-serif; font-size: 9.6pt; line-height: 1.28; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt; }
h1 + p { margin: 0 0 8pt; color: #444; font-size: 9pt; }
h2 { font-size: 11pt; margin: 9pt 0 3pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
p { margin: 0 0 4.5pt; text-align: justify; }
ol { margin: 0 0 4pt 16pt; padding: 0; } li { margin: 0 0 2pt; }
table { border-collapse: collapse; margin: 3pt 0 6pt; font-size: 8.8pt; }
th, td { border: 0.5pt solid #999; padding: 1.5pt 6pt; text-align: left; }
th { background: #eee; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt; }
strong { font-weight: 600; }
"""

def browser():
    for c in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium")):
        if c and Path(c).exists():
            return c
    sys.exit("no Edge/Chrome found")

body = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables"])
HTML.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style><body>{body}</body>", encoding="utf-8")
import tempfile, time
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:   # fresh profile: a running browser must not swallow the job
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-sandbox", f"--user-data-dir={profile}",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF}", str(HTML)],
                   check=True, capture_output=True, timeout=120)
    for _ in range(60):
        if PDF.exists() and PDF.stat().st_size > 0:
            break
        time.sleep(0.5)
HTML.unlink()

try:
    from pypdf import PdfReader
    print(f"{PDF.name}: {len(PdfReader(str(PDF)).pages)} page(s)")
except ImportError:
    print(f"wrote {PDF}")
