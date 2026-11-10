"""The generic pipeline reaches game engines only through ``games``.

pipeline-refactor.md §6.4: ``mcts/``, the training core in ``train/``,
``nn/__init__.py``, ``nn/policy_layout.py``, and ``nn/model_contract.py``
import nothing from the RSS engine (``core``, ``entities``, ``phases``) or
``g1889``; only ``games/`` does. RSS tooling and the model files are exempt.
Every import, including function-level and type-checking ones, counts.
At run time, games and model files load on first use, not at import.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PACKAGES = frozenset({"core", "entities", "phases", "g1889"})
RSS_TOOLING = frozenset({
    "train/analyze_game.py", "train/debug_trace.py", "train/strategy_data.py",
})
# (module, imported module) -> (imported names, why it may stay). Empty since
# step 4; a temporary entry names the step that removes it.
ALLOWED: dict[tuple[str, str], tuple[frozenset[str], str]] = {}


def _pipeline_modules() -> list[str]:
    paths = [*REPO_ROOT.glob("mcts/**/*.py"), *REPO_ROOT.glob("train/**/*.py")]
    paths += [REPO_ROOT / "nn" / name for name in ("__init__.py", "policy_layout.py", "model_contract.py")]
    modules = sorted(path.relative_to(REPO_ROOT).as_posix() for path in paths)
    return [module for module in modules if module not in RSS_TOOLING]


def _engine_imports(module: str) -> dict[str, set[str]]:
    """Imported engine module -> names, over every import in ``module``."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse((REPO_ROOT / module).read_text(), module)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ENGINE_PACKAGES:
                    found.setdefault(alias.name, set())
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in ENGINE_PACKAGES:
                found.setdefault(node.module, set()).update(a.name for a in node.names)
    return found


def test_pipeline_modules_cover_the_generic_pipeline():
    modules = _pipeline_modules()
    for module in ("mcts/search.py", "mcts/evaluator.py", "train/eval_server.py",
                   "train/replay_buffer.py", "train/trainer.py", "train/self_play.py",
                   "train/main.py", "train/config.py", "train/tournament.py", "nn/__init__.py"):
        assert module in modules
    assert not RSS_TOOLING & set(modules)


def test_generic_pipeline_imports_no_engine():
    found = {
        (module, imported): frozenset(names)
        for module in _pipeline_modules()
        for imported, names in _engine_imports(module).items()
    }
    unexpected = {
        key: sorted(names) for key, names in found.items()
        if key not in ALLOWED or not names <= ALLOWED[key][0]
    }
    assert not unexpected, f"engine imports outside games/: {unexpected}"
    # Keep the allowlist exact, so entries go when their imports do.
    stale = {key: sorted(ALLOWED[key][0] - found.get(key, frozenset())) for key in ALLOWED}
    assert not any(stale.values()), f"allowlisted imports no longer used: {stale}"


def test_importing_the_pipeline_loads_no_game_or_model():
    """Engines load with their game (``games.get_game``) and model files with
    their config (``nn.create_model``). So a 1889 process never loads RSS's
    engine, and ``games.rss`` -> ``nn.policy_layout`` -> ``nn`` is no cycle."""
    modules = [
        module.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        for module in _pipeline_modules() if not module.endswith("__main__.py")
    ]
    code = "\n".join((
        "import importlib, sys",
        f"for name in {modules!r}: importlib.import_module(name)",
        "from train.config import TrainingConfig",
        "TrainingConfig(), TrainingConfig(game='1889')",
        "print(sorted(name for name in sys.modules if name.split('.')[0] in "
        f"{sorted(ENGINE_PACKAGES)!r} or name.startswith(('games.', '_rss_model_impl_'))))",
    ))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines()[-1] == "[]"
