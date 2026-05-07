import pytest

from utils.gamelog_to_html import UnsupportedLogError, main, render_html


SAMPLE_GAMELOG = """# Self-Play Analysis: seed=42, 1600 simulations/move
# Noise: epsilon=0.15, dynamic alpha=5.0/K | Terminal blend: 0.75

Phase: INVEST  |  Turn: 1  |  CoO Level: 1  |  Active Player: 0  |  End Card: no

**Players**
  P0: $30 (NW $30) order=0 income=$0

**FI**: $4 income=$5

---

### Step 0: P0 [INVEST]

  NN Values: P0=+0.056, P1=+0.021, P2=-0.015
  NN Priors (top 1 of 5 legal):
     1.  30.8% (-4.5pp) \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588 AUCTION slot 3 (BPM, face $7)

  MCTS Visits (top 1, 1600 total):
     1.   720 (45.0%) Q=+0.126 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588 AUCTION slot 3 (BPM, face $7)
  A0GB Value: P0=-0.002, P1=+0.430, P2=-0.408

  **Action: AUCTION slot 3 (BPM, face $7)**

## Game Over

**Winner: P0 ($40)**
"""


def test_render_html_formats_full_analysis_log() -> None:
    rendered = render_html(
        SAMPLE_GAMELOG,
        source_name="gamelog.md",
        generated_at="2026-05-06 12:00 PDT",
    )

    assert "<title>Self-Play Analysis: seed=42, 1600 simulations/move</title>" in rendered
    assert "Rolling Stock Stars Analysis" in rendered
    assert 'class="state-panel"' in rendered
    assert 'href="#step-0-p0-invest"' in rendered
    assert "rank-row" in rendered
    assert "rank-bar-fill" in rendered
    assert "@media (max-width: 900px) and (prefers-color-scheme: dark)" in rendered
    assert "width: max(2px, 30.8%);" in rendered
    assert "\u2588" not in rendered
    assert 'class="chosen-action"' in rendered
    assert "<strong>Winner: P0 ($40)</strong>" in rendered


def test_render_html_formats_stats_only_rows_as_table() -> None:
    rendered = render_html(
        """# Self-Play MCTS Stats: seed=1, 20 simulations/move, batch=4
# Columns: step turn player phase | legal visited top gap eff depth(mean/max/greedy) batches avgB vb auto | action
000 T01 P0 INVEST         | legal= 5 vis= 5 top= 60.0% gap= 40.0pp eff= 2.3 depth= 1.4/ 4/ 2 batches=  3 avgB= 4.0 vb= 0 auto=0 | PASS (INVEST)
001 T01 P1 BID_IN_AUCTION | legal=10 vis= 8 top= 75.0% gap= 62.0pp eff= 1.8 depth= 2.1/ 5/ 3 batches=  4 avgB= 4.0 vb= 1 auto=0 | BID $5
""",
        source_name="stats.md",
        generated_at="2026-05-06 12:00 PDT",
    )

    assert 'class="stats-table"' in rendered
    assert "<td>000</td>" in rendered
    assert "<td>P1</td>" in rendered
    assert "<td>BID $5</td>" in rendered


def test_render_html_rejects_token_dump_logs() -> None:
    with pytest.raises(UnsupportedLogError, match="token-dump"):
        render_html("# Self-Play Analysis\n\n## Token Dump\n\nidx | token | width | values")


def test_main_writes_default_html_path(tmp_path) -> None:
    source = tmp_path / "gamelog.md"
    source.write_text(SAMPLE_GAMELOG, encoding="utf-8")

    assert main([str(source)]) == 0
    rendered = (tmp_path / "gamelog.html").read_text(encoding="utf-8")
    assert "Self-Play Analysis: seed=42" in rendered
