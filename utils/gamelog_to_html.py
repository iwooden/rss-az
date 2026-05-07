#!/usr/bin/env python3
"""Render ``train.analyze_game`` logs as a standalone static HTML page.

The analyzer output is Markdown-like, but it also contains dense preformatted
ranking rows, state dumps, and compact MCTS stats. This script keeps those
shapes readable without requiring a Markdown package.

Examples
--------
  .venv/bin/python utils/gamelog_to_html.py gamelog.md
  .venv/bin/python utils/gamelog_to_html.py gamelog.md -o public/index.html
  .venv/bin/python utils/gamelog_to_html.py gamelog.md --title "RSS Game 42"
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path


CSS = """
:root {
  color-scheme: only light;
  --bg: #f3f6f4;
  --paper: #ffffff;
  --ink: #1c2525;
  --muted: #5e6d69;
  --line: #d8e1dd;
  --line-strong: #b8c8c2;
  --accent: #166b72;
  --accent-soft: #e4f2f0;
  --amber: #9a6413;
  --amber-soft: #fff4d8;
  --green: #2d7351;
  --shadow: 0 16px 42px rgba(22, 32, 30, 0.08);
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}

a {
  color: inherit;
}

.hero {
  background: #17383a;
  color: #f9fbf7;
  padding: 2rem clamp(1rem, 4vw, 3rem);
  border-bottom: 5px solid #d79b32;
}

.hero-inner {
  max-width: 1440px;
  margin: 0 auto;
}

.eyebrow {
  margin: 0 0 0.45rem;
  color: #cddbd6;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  max-width: 72rem;
  font-size: clamp(1.75rem, 3vw, 3rem);
  line-height: 1.08;
  letter-spacing: 0;
}

.meta-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1.1rem;
  padding: 0;
  list-style: none;
}

.meta-grid li {
  padding: 0.35rem 0.55rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  color: #f7fbf7;
  font-size: 0.9rem;
}

.layout {
  display: grid;
  grid-template-columns: minmax(14rem, 20rem) minmax(0, 1fr);
  gap: 1.25rem;
  max-width: 1440px;
  margin: 0 auto;
  padding: 1.25rem;
}

.layout.no-toc {
  grid-template-columns: minmax(0, 1fr);
}

.toc {
  position: sticky;
  top: 1rem;
  align-self: start;
  max-height: calc(100vh - 2rem);
  overflow: auto;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.toc-title {
  padding: 0.85rem 1rem;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

.toc a {
  display: block;
  padding: 0.42rem 1rem;
  color: var(--ink);
  text-decoration: none;
  font-size: 0.9rem;
  border-bottom: 1px solid rgba(216, 222, 216, 0.6);
}

.toc a:hover {
  background: var(--accent-soft);
}

.toc .toc-l3 {
  padding-left: 1.55rem;
  color: var(--muted);
}

.content {
  min-width: 0;
}

.report-section,
.step-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  margin: 0 0 1rem;
  padding: 1rem;
}

.section-heading,
.step-title,
.subsection-heading {
  margin: 0 0 0.7rem;
  letter-spacing: 0;
  line-height: 1.2;
}

.section-heading {
  font-size: 1.35rem;
}

.step-title {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: baseline;
  color: #123f43;
  font-size: 1.15rem;
}

.subsection-heading {
  margin-top: 1rem;
  color: var(--accent);
  font-size: 1rem;
}

.state-panel {
  margin: 0.75rem 0;
  padding: 0.85rem 0.95rem;
  background: #fbfcf8;
  border-left: 4px solid var(--accent);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}

.state-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-bottom: 0.65rem;
}

.state-chip {
  display: inline-flex;
  gap: 0.25rem;
  align-items: baseline;
  padding: 0.25rem 0.45rem;
  background: var(--accent-soft);
  border: 1px solid #c7ddda;
  border-radius: 6px;
  font-size: 0.88rem;
}

.state-chip b {
  color: #24595d;
}

.state-subhead {
  margin: 0.75rem 0 0.25rem;
  color: var(--green);
  font-weight: 800;
}

.state-line,
.log-line {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.state-line {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.9rem;
}

.log-line {
  padding: 0.1rem 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.9rem;
}

.metric-line {
  color: #23474a;
  font-weight: 700;
}

.block-title {
  margin-top: 0.55rem;
  color: var(--accent);
  font-weight: 800;
}

.rank-row {
  display: grid;
  grid-template-columns: minmax(11rem, max-content) minmax(7rem, 14rem) minmax(12rem, 1fr);
  gap: 0.65rem;
  align-items: center;
  margin: 0.12rem 0;
  padding: 0.28rem 0.45rem;
  background: #f7faf7;
  border-left: 3px solid #cddfd7;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.9rem;
}

.rank-row.no-bar {
  display: block;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.rank-prefix {
  white-space: pre;
  color: #263d3b;
}

.rank-bar {
  display: block;
  width: 100%;
  height: 0.55rem;
  background-color: #dfe8e4;
  border: 1px solid #c7d7d0;
  border-radius: 999px;
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(22, 32, 30, 0.08);
  forced-color-adjust: none;
}

.rank-bar-fill {
  display: block;
  width: 0;
  height: 100%;
  background-color: #278173;
  background-image: linear-gradient(90deg, #278173, #d49a2e);
  border-radius: inherit;
  forced-color-adjust: none;
}

.rank-action {
  min-width: 0;
  overflow-wrap: anywhere;
}

.chosen-action {
  margin: 0.8rem 0;
  padding: 0.75rem 0.85rem;
  background: var(--amber-soft);
  border-left: 4px solid var(--amber);
  color: #3d2a08;
  font-weight: 800;
}

.auto-line {
  color: #5b4a17;
}

.turn-marker {
  margin: 1rem 0;
  padding: 0.5rem 0.75rem;
  background: #e9ede5;
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  color: #34433f;
  font-weight: 800;
}

hr {
  border: 0;
  border-top: 1px solid var(--line-strong);
  margin: 1rem 0;
}

.table-wrapper {
  width: 100%;
  overflow-x: auto;
  margin: 0.8rem 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
}

table {
  width: 100%;
  min-width: 42rem;
  border-collapse: collapse;
  font-size: 0.86rem;
}

th,
td {
  padding: 0.42rem 0.55rem;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
  text-align: left;
}

th {
  background: #edf3ef;
  color: #253331;
  font-weight: 800;
}

tr:last-child td {
  border-bottom: 0;
}

.align-right {
  text-align: right;
}

.align-center {
  text-align: center;
}

.stats-table td:nth-child(5),
.stats-table td:nth-child(6) {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}

.footer {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 1.25rem 2rem;
  color: var(--muted);
  font-size: 0.85rem;
}

@media (max-width: 900px) {
  .layout {
    display: block;
    padding: 0.9rem;
  }

  .toc {
    position: static;
    max-height: 16rem;
    margin-bottom: 1rem;
  }

  .hero {
    padding: 1.5rem 1rem;
  }

  .rank-row {
    grid-template-columns: minmax(0, 1fr) minmax(4.5rem, 6rem);
    gap: 0.35rem 0.55rem;
    font-size: 0.82rem;
  }

  .rank-action {
    grid-column: 1 / -1;
  }
}

@media (max-width: 900px) and (prefers-color-scheme: dark) {
  .rank-bar {
    background-color: rgba(201, 241, 231, 0.12);
    border-color: rgba(188, 245, 231, 0.58);
    box-shadow:
      inset 0 1px 2px rgba(0, 0, 0, 0.45),
      0 0 0 1px rgba(8, 47, 73, 0.35);
  }

  .rank-bar-fill {
    background-color: #7dd3fc;
    background-image: linear-gradient(90deg, #7dd3fc, #fef08a);
    box-shadow: 0 0 8px rgba(125, 211, 252, 0.55);
  }
}
"""


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    anchor: str


@dataclass
class RenderContext:
    used_anchors: dict[str, int]
    toc: list[TocEntry]

    def anchor_for(self, title: str) -> str:
        base = _slugify(title)
        count = self.used_anchors.get(base, 0)
        self.used_anchors[base] = count + 1
        if count:
            return f"{base}-{count + 1}"
        return base


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_RANK_RE = re.compile(r"^\s*\d+\.\s+")
_RANK_BAR_RE = re.compile(r"\s*(?P<bar>\u2588+)\s*")
_PERCENT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)%")
_NN_RANK_RE = re.compile(
    r"^(?P<prefix>\d+\.\s+[+-]?\d+(?:\.\d+)?%\s*(?:\([^)]+\))?)\s+"
    r"(?P<action>.*)$"
)
_MCTS_RANK_RE = re.compile(
    r"^(?P<prefix>\d+\.\s+\d+\s+\([^)]+%\)\s+Q=[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<action>.*)$"
)
_STATS_RE = re.compile(
    r"^(?P<step>\d{3})\s+T(?P<turn>\d{2})\s+P(?P<player>\d+)\s+"
    r"(?P<phase>[A-Z0-9_]+)\s+\|\s+(?P<metrics>.*?)\s+\|\s+(?P<action>.*)$"
)
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_TURN_RE = re.compile(r"^---\s+Turn\s+(\d+)\s+---$")
_TOKEN_DUMP_MARKERS = {"## Token Dump", "## Token Normalization Report"}


class UnsupportedLogError(ValueError):
    """Raised when the input log uses an analyzer mode this renderer ignores."""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _format_inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return _BOLD_RE.sub(r"<strong>\1</strong>", escaped)


def _escape_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _split_title(
    lines: list[str],
    *,
    fallback_title: str,
    override_title: str | None,
) -> tuple[str, list[str], list[str]]:
    metadata: list[str] = []
    title = override_title or fallback_title
    index = 0

    while index < len(lines) and lines[index].startswith("# "):
        text = lines[index][2:].strip()
        if index == 0 and override_title is None:
            title = text
        else:
            metadata.append(text)
        index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    return title, metadata, lines[index:]


def _metadata_items(metadata: list[str], source_name: str | None, generated_at: str) -> list[str]:
    items: list[str] = []
    for line in metadata:
        parts = [part.strip() for part in line.split(" | ") if part.strip()]
        items.extend(parts or [line])
    if source_name:
        items.append(f"Source: {source_name}")
    items.append(f"Generated: {generated_at}")
    return items


def _render_metadata(metadata: list[str], source_name: str | None, generated_at: str) -> str:
    items = _metadata_items(metadata, source_name, generated_at)
    if not items:
        return ""
    body = "\n".join(f"<li>{_format_inline(item)}</li>" for item in items)
    return f'<ul class="meta-grid">\n{body}\n</ul>'


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.split("|")]
    return len(cells) > 1 and all(_TABLE_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and _is_table_separator(lines[index + 1])
    )


def _split_table_row(line: str, columns: int) -> list[str]:
    cells = [cell.strip() for cell in line.split("|", max(0, columns - 1))]
    if len(cells) < columns:
        cells.extend("" for _ in range(columns - len(cells)))
    return cells[:columns]


def _alignment_classes(separator_cells: list[str], columns: int) -> list[str]:
    classes: list[str] = []
    for idx in range(columns):
        cell = separator_cells[idx].strip() if idx < len(separator_cells) else ""
        if cell.startswith(":") and cell.endswith(":"):
            classes.append("align-center")
        elif cell.endswith(":"):
            classes.append("align-right")
        else:
            classes.append("")
    return classes


def _render_table(lines: list[str], index: int) -> tuple[str, int]:
    headers = [cell.strip() for cell in lines[index].split("|")]
    separator_cells = [cell.strip() for cell in lines[index + 1].split("|")]
    columns = len(headers)
    align = _alignment_classes(separator_cells, columns)
    index += 2

    rows: list[list[str]] = []
    while index < len(lines):
        line = lines[index]
        if not line.strip() or _is_block_boundary(line):
            break
        if "|" not in line:
            break
        rows.append(_split_table_row(line, columns))
        index += 1

    head_cells = "".join(
        f'<th class="{_escape_attr(align[idx])}">{_format_inline(cell)}</th>'
        for idx, cell in enumerate(headers)
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td class="{_escape_attr(align[idx])}">{_format_inline(cell)}</td>'
            for idx, cell in enumerate(row)
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)
    return (
        '<div class="table-wrapper"><table>'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>",
        index,
    )


def _render_stats_table(lines: list[str], index: int) -> tuple[str, int]:
    rows: list[re.Match[str]] = []
    while index < len(lines):
        match = _STATS_RE.match(lines[index])
        if match is None:
            break
        rows.append(match)
        index += 1

    header = (
        "<thead><tr>"
        "<th>Step</th><th>Turn</th><th>Player</th><th>Phase</th>"
        "<th>Metrics</th><th>Action</th>"
        "</tr></thead>"
    )
    body_rows = []
    for match in rows:
        cells = [
            match.group("step"),
            f'T{match.group("turn")}',
            f'P{match.group("player")}',
            match.group("phase"),
            match.group("metrics"),
            match.group("action"),
        ]
        body_rows.append(
            "<tr>"
            + "".join(f"<td>{_format_inline(cell)}</td>" for cell in cells)
            + "</tr>"
        )
    body = "\n".join(body_rows)
    return (
        '<div class="table-wrapper"><table class="stats-table">'
        f"{header}<tbody>{body}</tbody></table></div>",
        index,
    )


def _render_phase_header(line: str) -> str:
    chips = []
    for part in line.split("|"):
        part = part.strip()
        if ":" in part:
            key, value = part.split(":", 1)
            chips.append(
                '<span class="state-chip">'
                f"<b>{_format_inline(key.strip())}</b>"
                f"<span>{_format_inline(value.strip())}</span>"
                "</span>"
            )
        else:
            chips.append(f'<span class="state-chip">{_format_inline(part)}</span>')
    return '<div class="state-chips">' + "\n".join(chips) + "</div>"


def _render_state_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("**") and stripped.endswith("**") and stripped.count("**") == 2:
        return f'<div class="state-subhead">{_format_inline(stripped[2:-2])}</div>'
    if stripped.startswith("**") and "**:" in stripped:
        return f'<div class="state-line">{_format_inline(stripped)}</div>'
    return f'<div class="state-line">{_format_inline(line)}</div>'


def _render_state_panel(lines: list[str], index: int) -> tuple[str, int]:
    parts = ['<section class="state-panel">', _render_phase_header(lines[index])]
    index += 1

    while index < len(lines):
        line = lines[index]
        if _is_block_boundary(line) or line.startswith("Phase: "):
            break
        rendered = _render_state_line(line)
        if rendered:
            parts.append(rendered)
        index += 1

    parts.append("</section>")
    return "\n".join(parts), index


def _is_block_boundary(line: str) -> bool:
    stripped = line.strip()
    return (
        bool(_HEADING_RE.match(line))
        or stripped == "---"
        or bool(_TURN_RE.match(stripped))
        or bool(_STATS_RE.match(line))
    )


def _line_class(line: str) -> str:
    stripped = line.strip()
    classes = ["log-line"]
    if _RANK_RE.match(line):
        classes.append("rank-row")
    if stripped.startswith(("NN Values:", "A0GB Value:", "Terminal values")):
        classes.append("metric-line")
    if stripped.startswith(("NN Priors", "MCTS Visits")):
        classes.append("block-title")
    if stripped.startswith("\u21b3 auto:"):
        classes.append("auto-line")
    return " ".join(classes)


def _rank_width_percent(text: str) -> float | None:
    match = _PERCENT_RE.search(text)
    if match is None:
        return None
    return max(0.0, min(100.0, float(match.group(1))))


def _split_rank_line(text: str) -> tuple[str, str]:
    for pattern in (_MCTS_RANK_RE, _NN_RANK_RE):
        match = pattern.match(text)
        if match is not None:
            return match.group("prefix").rstrip(), match.group("action").strip()
    return text, ""


def _render_rank_line(line: str) -> str:
    stripped = line.strip()
    bar_match = _RANK_BAR_RE.search(stripped)
    if bar_match is not None:
        prefix = stripped[:bar_match.start()].rstrip()
        action = stripped[bar_match.end():].strip()
    else:
        prefix, action = _split_rank_line(stripped)

    width_percent = _rank_width_percent(stripped)
    if width_percent is None or not action:
        return f'<div class="rank-row no-bar">{_format_inline(stripped)}</div>'

    width = "0%" if width_percent == 0.0 else f"max(2px, {width_percent:.1f}%)"
    style = _escape_attr(f"width: {width};")
    label = _escape_attr(f"{width_percent:.1f}%")
    return (
        '<div class="rank-row">'
        f'<span class="rank-prefix">{_format_inline(prefix)}</span>'
        f'<span class="rank-bar" role="img" aria-label="{label}">'
        f'<span class="rank-bar-fill" style="{style}"></span>'
        "</span>"
        f'<span class="rank-action">{_format_inline(action)}</span>'
        "</div>"
    )


def _render_log_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("**Action:"):
        return f'<div class="chosen-action">{_format_inline(stripped)}</div>'
    if _RANK_RE.match(line):
        return _render_rank_line(line)
    return f'<div class="{_escape_attr(_line_class(line))}">{_format_inline(line)}</div>'


def _render_heading(
    hashes: str,
    title: str,
    ctx: RenderContext,
    *,
    add_to_toc: bool,
) -> str:
    level = min(max(len(hashes), 2), 4)
    tag = f"h{level}"
    if level == 2:
        cls = "section-heading"
    else:
        cls = "subsection-heading"
    anchor = ctx.anchor_for(title)
    if add_to_toc:
        ctx.toc.append(TocEntry(level=level, title=title, anchor=anchor))
    return f'<{tag} id="{_escape_attr(anchor)}" class="{cls}">{_format_inline(title)}</{tag}>'


def _render_blocks(lines: list[str]) -> tuple[str, list[TocEntry]]:
    ctx = RenderContext(used_anchors={}, toc=[])
    parts: list[str] = []
    open_step = False
    open_report = False
    index = 0

    def close_step() -> None:
        nonlocal open_step
        if open_step:
            parts.append("</section>")
            open_step = False

    def close_report() -> None:
        nonlocal open_report
        if open_report:
            parts.append("</section>")
            open_report = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if _is_table_start(lines, index):
            table_html, index = _render_table(lines, index)
            parts.append(table_html)
            continue

        if _STATS_RE.match(line):
            stats_html, index = _render_stats_table(lines, index)
            parts.append(stats_html)
            continue

        turn_match = _TURN_RE.match(stripped)
        if turn_match is not None:
            title = f"Turn {turn_match.group(1)}"
            anchor = ctx.anchor_for(title)
            ctx.toc.append(TocEntry(level=2, title=title, anchor=anchor))
            parts.append(f'<div id="{_escape_attr(anchor)}" class="turn-marker">{_format_inline(title)}</div>')
            index += 1
            continue

        if stripped == "---":
            parts.append("<hr>")
            index += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match is not None:
            hashes, heading_title = heading_match.groups()
            is_step = heading_title.startswith("Step ")
            opens_report = len(hashes) <= 2 and heading_title != "Token Dump"

            if is_step:
                close_step()
                close_report()
                anchor = ctx.anchor_for(heading_title)
                ctx.toc.append(TocEntry(level=3, title=heading_title, anchor=anchor))
                parts.append(f'<section id="{_escape_attr(anchor)}" class="step-card">')
                parts.append(f'<h2 class="step-title">{_format_inline(heading_title)}</h2>')
                open_step = True
            elif opens_report:
                close_step()
                close_report()
                anchor = ctx.anchor_for(heading_title)
                ctx.toc.append(TocEntry(level=2, title=heading_title, anchor=anchor))
                parts.append(f'<section id="{_escape_attr(anchor)}" class="report-section">')
                parts.append(f'<h2 class="section-heading">{_format_inline(heading_title)}</h2>')
                open_report = True
            else:
                add_to_toc = heading_title != "Token Dump"
                parts.append(_render_heading(hashes, heading_title, ctx, add_to_toc=add_to_toc))
            index += 1
            continue

        if line.startswith("Phase: "):
            state_html, index = _render_state_panel(lines, index)
            parts.append(state_html)
            continue

        parts.append(_render_log_line(line))
        index += 1

    close_step()
    close_report()
    return "\n".join(parts), ctx.toc


def _render_toc(toc: list[TocEntry]) -> str:
    if not toc:
        return ""
    links = "\n".join(
        f'<a class="toc-l{entry.level}" href="#{_escape_attr(entry.anchor)}">'
        f"{_format_inline(entry.title)}</a>"
        for entry in toc
    )
    return (
        '<nav class="toc" aria-label="Log navigation">'
        '<div class="toc-title">Contents</div>'
        f"{links}</nav>"
    )


def _format_timestamp(value: dt.datetime | None = None) -> str:
    timestamp = value or dt.datetime.now().astimezone()
    return timestamp.strftime("%Y-%m-%d %H:%M %Z")


def render_html(
    text: str,
    *,
    source_name: str | None = None,
    title: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Return standalone HTML for one analyzer log."""
    lines = text.splitlines()
    unsupported_markers = _TOKEN_DUMP_MARKERS.intersection(line.strip() for line in lines)
    if unsupported_markers:
        markers = ", ".join(sorted(unsupported_markers))
        raise UnsupportedLogError(
            "token-dump analyzer output is not supported by this renderer "
            f"(found {markers})"
        )
    fallback_title = Path(source_name).stem if source_name else "RSS Game Log"
    page_title, metadata, body_lines = _split_title(
        lines,
        fallback_title=fallback_title,
        override_title=title,
    )
    generated_at = generated_at or _format_timestamp()
    body_html, toc = _render_blocks(body_lines)
    toc_html = _render_toc(toc)
    layout_class = "layout" if toc else "layout no-toc"
    metadata_html = _render_metadata(metadata, source_name, generated_at)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(page_title)}</title>\n"
        f"<style>\n{CSS}\n</style>\n"
        "</head>\n"
        "<body>\n"
        '<header class="hero"><div class="hero-inner">\n'
        '<p class="eyebrow">Rolling Stock Stars Analysis</p>\n'
        f"<h1>{_format_inline(page_title)}</h1>\n"
        f"{metadata_html}\n"
        "</div></header>\n"
        f'<div class="{layout_class}">\n'
        f"{toc_html}\n"
        f'<main class="content">\n{body_html}\n</main>\n'
        "</div>\n"
        '<footer class="footer">Generated by utils/gamelog_to_html.py.</footer>\n'
        "</body>\n"
        "</html>\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render train.analyze_game output as standalone HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  .venv/bin/python utils/gamelog_to_html.py gamelog.md\n"
            "  .venv/bin/python utils/gamelog_to_html.py gamelog.md -o public/index.html\n"
            "  .venv/bin/python utils/gamelog_to_html.py gamelog.md --stdout > gamelog.html"
        ),
    )
    parser.add_argument("input", type=Path, help="analyze_game output file, usually .md")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="HTML output path (default: input path with .html suffix)",
    )
    parser.add_argument("--title", default=None, help="override the page title")
    parser.add_argument("--stdout", action="store_true", help="write HTML to stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: input path is not a file: {args.input}", file=sys.stderr)
        return 2

    text = args.input.read_text(encoding="utf-8", errors="replace")
    try:
        rendered = render_html(
            text,
            source_name=str(args.input),
            title=args.title,
        )
    except UnsupportedLogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.stdout:
        sys.stdout.write(rendered)
        return 0

    output = args.output or args.input.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
