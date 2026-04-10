#!/usr/bin/env python3
"""Extract Python-interaction lines from Cython HTML annotations.

Cython's ``-a`` flag (or ``annotate=True`` in cythonize) produces an HTML
report where each source line is coloured by how much Python/C-API work it
generates.  White lines are pure C; progressively deeper yellow means more
Python interaction.  This script parses those reports and prints the yellow
lines so you can audit hot-path overhead without opening a browser.

Accepts ``.html`` files directly, or ``.pyx`` files — for the latter it
looks for a sibling ``.html`` and, if missing, generates one with
``cython -a``.  Glob patterns (``core/*.pyx``) work too.

Examples
--------
  # All yellow lines in one module:
  python utils/cython-html.py core/actions.html

  # Only the worst offenders, with context:
  python utils/cython-html.py core/actions.pyx --min-score 20 -C 2

  # Quick scan of every compiled module:
  python utils/cython-html.py core/*.pyx entities/*.pyx phases/*.pyx

  # Sort by score (worst first):
  python utils/cython-html.py core/actions.pyx --sort

  # CI gate — exit 1 if any line exceeds threshold:
  python utils/cython-html.py core/actions.pyx --min-score 50 --fail
"""

from __future__ import annotations

import argparse
import glob
import html
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

# Each source line in the annotation HTML looks like:
#   <pre class="cython line score-N" onclick="...">+<span>LINE</span>: SRC</pre>
# Score 0 = pure C (white), score > 0 = Python interaction (yellow).
_LINE_RE = re.compile(
    r'<pre\s+class=["\']cython line score-(\d+)["\']'  # score
    r'[^>]*>'                                           # rest of tag attrs
    r'[+\xad\s]*'                                       # optional +/- toggle
    r'<span[^>]*>(\d+)</span>'                          # line number
    r':\s?(.*?)</pre>',                                 # source text
)
_TAG_RE = re.compile(r'<[^>]+>')


def _strip_tags(s: str) -> str:
    return html.unescape(_TAG_RE.sub('', s))


def parse_annotation(path: Path) -> list[tuple[int, int, str]]:
    """Return ``[(score, lineno, source_text), ...]`` for every source line."""
    text = path.read_text(encoding='utf-8', errors='replace')
    return [
        (int(m.group(1)), int(m.group(2)), _strip_tags(m.group(3)))
        for m in _LINE_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# .pyx -> .html resolution
# ---------------------------------------------------------------------------

def resolve_html(path: Path) -> Path | None:
    """Given a ``.pyx`` or ``.html`` path, return the annotation HTML.

    For ``.pyx`` inputs, looks for a sibling ``.html``.  If absent, runs
    ``cython -a`` to generate it.  Returns ``None`` on failure.
    """
    if path.suffix == '.html':
        return path if path.exists() else None

    html_path = path.with_suffix('.html')
    if html_path.exists():
        return html_path

    # Generate annotation HTML via cython -a.
    try:
        subprocess.run(
            [sys.executable, '-m', 'cython', '-a', str(path)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"warning: failed to generate {html_path}: {exc}",
              file=sys.stderr)
        return None
    return html_path if html_path.exists() else None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_file_report(
    path: Path,
    lines: list[tuple[int, int, str]],
    *,
    min_score: int,
    context: int,
    sort_by_score: bool,
) -> int:
    """Print yellow lines for one file.  Returns the count of yellow lines."""
    yellow_indices = [i for i, (s, _, _) in enumerate(lines) if s >= min_score]
    if not yellow_indices:
        return 0

    yellow_count = len(yellow_indices)
    max_score = max(s for s, _, _ in lines)

    print(f"# {path}: {yellow_count} yellow line{'s' if yellow_count != 1 else ''} "
          f"(score >= {min_score}), "
          f"{len(lines)} total, max score {max_score}")
    print()

    if sort_by_score:
        # Show yellow lines only, sorted by descending score.
        for i in sorted(yellow_indices, key=lambda i: lines[i][0], reverse=True):
            score, lineno, source = lines[i]
            print(f"  [{score:3d}] {lineno:4d}: {source}")
    else:
        # Source-order display with optional context.
        show: set[int] = set()
        for i in yellow_indices:
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                show.add(j)
        prev_idx = -2
        for i in sorted(show):
            score, lineno, source = lines[i]
            if i > prev_idx + 1:
                print("  ---")
            marker = f"[{score:3d}]" if score >= min_score else "     "
            print(f"  {marker} {lineno:4d}: {source}")
            prev_idx = i

    # Score distribution.
    buckets: dict[str, int] = {}
    for score, _, _ in lines:
        if score == 0:
            continue
        if score <= 5:
            key = " 1-5 "
        elif score <= 15:
            key = " 6-15"
        elif score <= 50:
            key = "16-50"
        else:
            key = "  51+"
        buckets[key] = buckets.get(key, 0) + 1

    if buckets:
        print()
        print("  Score distribution:", end="")
        for key in [" 1-5 ", " 6-15", "16-50", "  51+"]:
            if key in buckets:
                print(f"  [{key.strip():>5s}]={buckets[key]}", end="")
        print()

    print()
    return yellow_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'files', nargs='+', metavar='FILE',
        help='.pyx or .html files (globs expanded)',
    )
    ap.add_argument(
        '--min-score', type=int, default=1,
        help='minimum score to report (default: 1, i.e. any yellow)',
    )
    ap.add_argument(
        '--context', '-C', type=int, default=0,
        help='show N surrounding white lines for context',
    )
    ap.add_argument(
        '--sort', action='store_true',
        help='sort lines by descending score instead of source order',
    )
    ap.add_argument(
        '--fail', action='store_true',
        help='exit 1 if any yellow lines found (useful as CI gate)',
    )
    args = ap.parse_args(argv)

    # Expand globs (shell may not expand on Windows).
    paths: list[Path] = []
    for pattern in args.files:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            paths.extend(Path(p) for p in sorted(expanded))
        else:
            paths.append(Path(pattern))

    total_yellow = 0
    files_processed = 0

    for path in paths:
        html_path = resolve_html(path)
        if html_path is None:
            print(f"warning: skipping {path} (no annotation HTML)",
                  file=sys.stderr)
            continue

        lines = parse_annotation(html_path)
        if not lines:
            print(f"warning: no source lines found in {html_path}",
                  file=sys.stderr)
            continue

        files_processed += 1
        total_yellow += print_file_report(
            path,
            lines,
            min_score=args.min_score,
            context=args.context,
            sort_by_score=args.sort,
        )

    if files_processed > 1:
        print(f"# Total: {total_yellow} yellow line{'s' if total_yellow != 1 else ''} "
              f"across {files_processed} files")

    if args.fail and total_yellow > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
