import asyncio
import os
import subprocess
import sys
from pathlib import Path
from playwright.async_api import async_playwright

def run_nbconvert():
    print("[*] Converting final_report.ipynb to HTML...")
    # Find python interpreter
    python_exe = sys.executable
    
    # Run nbconvert to html (no-input to keep it clean)
    cmd = [
        python_exe, "-m", "jupyter", "nbconvert",
        "--to", "html",
        "--no-input",
        "notebooks/final_report.ipynb"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[!] nbconvert failed:")
        print(result.stderr)
        sys.exit(1)
    print("[*] HTML conversion successful.")

async def convert_html_to_pdf():
    print("[*] Starting Playwright PDF engine...")
    html_path = Path("notebooks/final_report.html").resolve()
    pdf_path = Path("notebooks/final_report.pdf").resolve()
    
    if not html_path.exists():
        print(f"[!] HTML file not found at: {html_path}")
        sys.exit(1)
        
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load the local HTML file
        print(f"[*] Loading: file:///{html_path}")
        await page.goto(f"file:///{html_path}")
        
        # Wait for MathJax (LaTeX formula) and images to render completely
        print("[*] Waiting for assets to render...")
        await page.wait_for_timeout(3000)
        
        # Print to PDF with print_background active to keep code highlighting & maps intact
        print(f"[*] Printing to: {pdf_path}")
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"}
        )
        await browser.close()
    print("[*] PDF export complete!")

if __name__ == "__main__":
    # Force the correct Windows event loop policy to support subprocesses
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    run_nbconvert()
    asyncio.run(convert_html_to_pdf())
