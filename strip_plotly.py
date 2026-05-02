#!/usr/bin/env python3
"""Strip Plotly JS bundle from plot HTML files.

Removes all <script> tags that are the Plotly library bundle
from all *.html files excluding index.html.
Keeps only the plot div and the Plotly.newPlot() initialization script.
"""

import glob
import os
from pathlib import Path

from bs4 import BeautifulSoup


def main():
    repo_root = Path(__file__).parent
    html_files = glob.glob(str(repo_root / "**" / "*.html"), recursive=True)
    html_files = [f for f in html_files if os.path.basename(f) != "index.html"]

    patched = 0
    already_clean = 0

    for filepath in html_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")
        scripts = soup.find_all("script")

        removed_any = False
        for script in scripts:
            if not script.string:
                # Script with src attribute, remove it
                script.decompose()
                removed_any = True
                continue

            script_content = script.string

            # Keep PlotlyConfig (tiny config script)
            if "window.PlotlyConfig" in script_content:
                continue

            # Remove Plotly library (contains "plotly.js v" in header comment)
            if "plotly.js v" in script_content or "Plotly.js v" in script_content:
                script.decompose()
                removed_any = True
                continue

            # Keep initialization scripts (contain Plotly.newPlot( or Plotly.react( )
            if "Plotly.newPlot(" in script_content or "Plotly.react(" in script_content:
                continue

            # Remove everything else
            script.decompose()
            removed_any = True

        if removed_any:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(soup))
            patched += 1
        else:
            already_clean += 1

    print(f"Files patched: {patched}")
    print(f"Already clean: {already_clean}")


if __name__ == "__main__":
    main()
