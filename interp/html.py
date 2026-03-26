"""Shared HTML report utilities for interpretability reports.

Provides a dark-themed HTML page wrapper, reusable CSS modules, JavaScript
helpers, and a browser file-opener — eliminating duplication across the
11 interp report generators.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# CSS modules
# ---------------------------------------------------------------------------

_BASE_CSS_TEMPLATE = """\
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  background: #1a1a2e; color: #e0e0e0;
  margin: 2rem auto; max-width: __MAX_WIDTH__px; padding: 0 1rem;
}
h1 { color: #f0f0f0; font-size: 1.4rem; margin-bottom: 0.3rem; }
h2 { color: #ccc; font-size: 1.1rem; margin-top: 2rem;
      border-bottom: 1px solid #333; padding-bottom: 0.3rem; }
h3 { color: #aaa; font-size: 0.95rem; margin-top: 1rem; }
.meta { color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }
table {
  border-collapse: collapse; width: 100%;
  font-size: 0.82rem; margin-bottom: 1.5rem;
}
th, td { padding: 5px 8px; border: 1px solid #2a2a4a; text-align: right; }
th { background: #16213e; color: #aaa; font-weight: 600;
     position: sticky; top: 0; z-index: 1; }
th:first-child, td:first-child { text-align: left; }
td:first-child {
  font-family: "SF Mono", "Fira Code", Consolas, monospace;
  font-size: 0.8rem; color: #ccc;
}
tr:hover td { border-color: #555; }"""

BAR_CSS = """\
.bar-container { display: inline-block; width: 120px; vertical-align: middle; }
.bar {
  display: inline-block; height: 12px; border-radius: 2px;
  vertical-align: middle;
}
.bar-blue { background: #4a9eff; }
.bar-green { background: #4ecca3; }
.bar-orange { background: #e9a945; }
.bar-red { background: #e94560; }
.bar-yellow { background: #e9a945; }"""

TAG_CSS = """\
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 3px;
  font-size: 0.75rem; font-weight: 600;
}
.tag-ok { background: #1a3a2a; color: #4ecca3; }
.tag-warn { background: #3a2a1a; color: #e9a945; }
.tag-bad { background: #3a1a1a; color: #e94560; }"""

STAT_BOX_CSS = """\
.stat-box {
  display: inline-block; background: #16213e; border: 1px solid #2a2a4a;
  border-radius: 4px; padding: 8px 16px; margin: 4px 8px 4px 0;
  font-size: 0.85rem;
}
.stat-label { color: #888; font-size: 0.75rem; }
.stat-value { color: #e0e0e0; font-size: 1.1rem; font-weight: 600; }"""

HIST_BAR_CSS = """\
.hist-bar {
  display: inline-block; background: #4a9eff; height: 16px;
  border-radius: 1px; vertical-align: middle;
}"""

# ---------------------------------------------------------------------------
# JavaScript utilities
# ---------------------------------------------------------------------------

JS_MAKE_BAR = """\
function makeBar(val, maxVal, cls) {
  const pct = maxVal > 0 ? Math.min(val / maxVal * 100, 100) : 0;
  return '<span class="bar-container"><span class="bar ' + cls + '" style="width:' + pct + '%"></span></span>';
}"""

JS_FMT_PCT = "function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }"
JS_FMT_VAL = "function fmtVal(v, d) { return v.toFixed(d === undefined ? 3 : d); }"


# ---------------------------------------------------------------------------
# HTML page wrapper
# ---------------------------------------------------------------------------


def html_page(
    title: str,
    *,
    meta: str = "",
    body: str = "",
    script: str = "",
    extra_css: str = "",
    max_width: int = 1100,
) -> str:
    """Build a self-contained HTML report page with the standard dark theme.

    Parameters
    ----------
    title : str
        Page title (used in both ``<title>`` and ``<h1>``).
    meta : str
        Content for the ``<div class="meta">`` subtitle line.
    body : str
        HTML body content (section headings, tables, divs).
    script : str
        JavaScript code (without ``<script>`` tags).
    extra_css : str
        Additional CSS rules appended after the base stylesheet.
    max_width : int
        Maximum page width in pixels (default 1100).
    """
    css = _BASE_CSS_TEMPLATE.replace("__MAX_WIDTH__", str(max_width))
    if extra_css:
        css += "\n" + extra_css

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        f"<title>{title}</title>",
        "<style>",
        css,
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]
    if meta:
        parts.append(f'<div class="meta">{meta}</div>')
    if body:
        parts.append(body)
    if script:
        parts.append("<script>")
        parts.append(script)
        parts.append("</script>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Browser file opener
# ---------------------------------------------------------------------------


def open_file(path: Path) -> None:
    """Open a file with the platform's default handler."""
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(
                ["open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Windows":
            subprocess.Popen(
                ["start", "", str(path)],
                shell=True,  # noqa: S602
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except OSError:
        print(f"  Could not open browser. Open manually: {path}")
