"""The tools of the first 1889 run (1889/model-design.md §11): the epoch-20 rescale
and the 8.5 profile, here on the CPU with what needs the 5090 stubbed or skipped;
the startup banner and ``--help``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from train.config import TrainingConfig
from train.logging import TrainingLogger
from train.main import _build_parser


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path("cloud") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rescale = _load_script("rescale_1889")
profile = _load_script("profile_1889")


# ---------------------------------------------------------------------------
# The rescale
# ---------------------------------------------------------------------------

def _write_logs(logdir: Path, epochs: dict[int, tuple[float, float, float, float]]) -> None:
    """Self-play's scalars: each epoch's game lengths by player count, and its samples."""
    with SummaryWriter(str(logdir)) as writer:
        for epoch, (three, four, five, samples) in epochs.items():
            writer.add_scalar("self_play_3p/game_length_mean", three, epoch)
            writer.add_scalar("self_play_4p/game_length_mean", four, epoch)
            writer.add_scalar("self_play_5p/game_length_mean", five, epoch)
            writer.add_scalar("self_play_aggregate/total_examples", samples, epoch)


def test_the_rules_reproduce_the_first_configs_values() -> None:
    """Section 7.1's planning lengths and samples give 1889-v0.json's windows."""
    config = TrainingConfig.from_json(Path("train_configs/1889-v0.json").read_text())
    new = rescale.rescale({3: 575.0, 4: 570.0, 5: 650.0}, 300_000.0, config.batch_size)
    assert new["temp_anneal_starts"] == config.temp_anneal_starts == [100, 100, 115]
    assert new["temp_anneal_ends"] == config.temp_anneal_ends == [200, 200, 230]
    assert new["policy_target_temp_anneal_starts"] == config.policy_target_temp_anneal_starts
    assert new["policy_target_temp_anneal_ends"] == config.policy_target_temp_anneal_ends
    # The config rounds up to round numbers; the rule itself is within 1%.
    assert new["buffer_capacity"] == 990_000
    assert new["training_steps_per_epoch"] == 1_290


def test_measure_reads_the_window_through_the_epoch_asked_for() -> None:
    samples = [(e, 100.0 * e) for e in range(1, 11)]
    scalars = {
        "self_play_aggregate/total_examples": samples,
        **{f"self_play_{n}p/game_length_mean": [(e, float(n * e)) for e in range(1, 11)] for n in (3, 4, 5)},
    }
    epochs, lengths, mean = rescale.measure(scalars, range(3, 6), epoch=None, window=4)
    assert epochs == [7, 8, 9, 10]
    assert lengths == {3: 25.5, 4: 34.0, 5: 42.5}
    assert mean == 850.0
    epochs, _, mean = rescale.measure(scalars, range(3, 6), epoch=5, window=4)
    assert epochs == [2, 3, 4, 5]
    assert mean == 350.0
    # A short run reads what it has.
    assert rescale.measure(scalars, range(3, 6), epoch=2, window=4)[0] == [1, 2]
    with pytest.raises(SystemExit, match="no epochs through 0"):
        rescale.measure(scalars, range(3, 6), epoch=0, window=4)
    del scalars["self_play_4p/game_length_mean"]
    with pytest.raises(SystemExit, match="self_play_4p/game_length_mean"):
        rescale.measure(scalars, range(3, 6), epoch=None, window=4)


def test_the_helper_prints_the_changes_and_writes_nothing(tmp_path, capsys) -> None:
    logdir = tmp_path / "runs"
    _write_logs(logdir, {e: (800.0, 900.0, 1000.0, 400_000.0) for e in range(1, 21)})
    before = sorted(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    rescale.main(["--config", "train_configs/1889-v0.json", "--logdir", str(logdir)])
    output = capsys.readouterr().out
    assert sorted(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()) == before
    assert "epochs 16-20 (5 averaged)" in output
    assert "Samples per epoch: 400,000" in output
    assert '"temp_anneal_starts": [140, 160, 175]' in output
    assert '"temp_anneal_ends": [280, 320, 350]' in output
    assert '"buffer_capacity": 1320000' in output
    assert '"training_steps_per_epoch": 1720' in output
    assert "1000000" in output and "[100, 100, 115]" in output  # the old values
    assert "python -m train --config train_configs/1889-v0.json --resume latest" in output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--config", "nowhere.json"], "no config file"),
        (["--logdir", "nowhere"], "no TensorBoard events"),
    ],
)
def test_the_helper_says_what_is_missing(argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        rescale.main(argv)


# ---------------------------------------------------------------------------
# The profile's checks
# ---------------------------------------------------------------------------

def _statuses(checks: list[dict[str, str]]) -> dict[str, str]:
    return {check["name"]: check["status"] for check in checks}


def _difference(logit_max: float = 0.001) -> dict[str, float]:
    return {
        "logit_count": 10, "logit_max_abs": logit_max, "logit_rms": 0.0, "logit_outside_tolerance": 0,
        "value_count": 10, "value_max_abs": 0.0001, "value_rms": 0.0, "value_outside_tolerance": 0,
        "policy_max_tv": 0.001,
    }


def _pair(ms: list[float], recorded: list[int]) -> dict[str, object]:
    return profile.summarize_pair(ms, recorded)


def _passing_sections() -> dict[str, dict[str, object]]:
    pairs = {
        "B=8,K=256": _pair([9.0, 3.0, 3.0, 3.0], [1, 0, 0, 0]),
        "B=8,K=512": _pair([8.0, 9.0, 3.0, 3.0], [0, 1, 0, 0]),
    }
    once, histogram = profile.launches_recorded_once(pairs)
    return {
        "model_time": {
            "rows": [{"case": "v0", "batch": 256, "gpu_ms": 8.0}, {"case": "v3", "batch": 256, "gpu_ms": 4.4}],
            "ratios": {"256": {"v0_over_v3": 1.82, "padded_three_tensors_over_v0": 1.2, "8heads_over_v0": 1.1}},
            "padded_matches_v0": _difference(0.0),
        },
        "compile": {"graph_breaks": {}, "unique_graphs": 2},
        "launches": {
            "pairs": pairs,
            "compiles_after_warmup": 0,
            "recorded_once_each": once,
            "record_launch_histogram": histogram,
            "graph_trees": {"recorded_nodes": 2},
            "against_eager": _difference(),
        },
        "replay": {
            "against_eager": _difference(),
            "filler_invariance": _difference(0.0),
            "weight_sync": {"replay_against_eager": _difference(), "replay_against_before": _difference(0.5)},
        },
        "memory": {
            "samples": [{}], "launches": 7200, "new_captures": 0, "new_compiles": 0,
            "growth_mb": {"reserved_mb": 0.0, "allocated_mb": 0.0, "device_free_mb": -1.0},
        },
        "train_steps": {"batch": 256, "shapes": {"256x15": {"steps_per_second": 20.0, "hours_for_40k_steps": 0.5}}},
    }


def test_a_good_run_passes_every_check() -> None:
    checks = profile.build_checks(_passing_sections(), profile.CUDA_MODE)
    statuses = _statuses(checks)
    assert statuses["v0/v3 model time, batch 256"] == "PASS"
    assert statuses["each (B, K) records one graph"] == "PASS"
    assert statuses["recorded graphs against pairs"] == "PASS"
    assert "FAIL" not in statuses.values()


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        (lambda s: s["model_time"]["ratios"]["256"].update(v0_over_v3=2.3), "v0/v3 model time, batch 256"),
        (lambda s: s["model_time"]["ratios"]["256"].update(v0_over_v3=1.5), "v0/v3 model time, batch 256"),
        (lambda s: s["model_time"]["ratios"]["256"].update(padded_three_tensors_over_v0=0.9),
         "v0's bias path against padded, batch 256"),
        (lambda s: s["compile"].update(graph_breaks={"a": 1}), "graph breaks in warmup"),
        (lambda s: s["compile"].update(unique_graphs=3), "compiled graphs after warmup"),
        (lambda s: s["launches"].update(compiles_after_warmup=1), "recompiles across launches"),
        (lambda s: s["launches"].update(recorded_once_each=False), "each (B, K) records one graph"),
        (lambda s: s["launches"].update(graph_trees={"recorded_nodes": 3}), "recorded graphs against pairs"),
        (lambda s: s["launches"]["pairs"].update({"B=8,K=512": _pair([3.0, 30.0, 3.0, 30.0, 3.0, 30.0], [1, 0, 0, 0, 0, 0])}),
         "latency settles at every (B, K)"),
        (lambda s: s["launches"].update(against_eager=_difference(0.5)), "compiled against eager, launches 1-3"),
        (lambda s: s["replay"].update(against_eager=_difference(0.5)), "compiled against eager, replay"),
        (lambda s: s["replay"].update(filler_invariance=_difference(0.5)), "filler changes nothing, across K"),
        (lambda s: s["replay"]["weight_sync"].update(replay_against_before=_difference(0.0)),
         "replay follows a weight sync"),
        (lambda s: s["memory"]["growth_mb"].update(reserved_mb=1024.0), "memory over 7,200 launches"),
        (lambda s: s["memory"].update(new_captures=4), "memory over 7,200 launches"),
        (lambda s: s["train_steps"]["shapes"]["256x15"].update(steps_per_second=1.0), "probe training, 256x15"),
        (lambda s: s.update(compile={"status": "error", "error": "boom"}), "compile"),
    ],
)
def test_each_bad_result_fails_its_check(change, failing: str) -> None:
    sections = _passing_sections()
    change(sections)
    statuses = _statuses(profile.build_checks(sections, profile.CUDA_MODE))
    assert statuses[failing] == "FAIL"


def test_a_pair_records_and_settles_by_launch_number() -> None:
    pair = profile.summarize_pair([50.0, 80.0, 5.0, 5.0, 5.0, 6.0], [0, 1, 0, 0, 0, 0])
    assert pair["record_launches"] == [2]
    assert pair["settled"]
    assert pair["settled_ms"] == 5.0
    # One spike isn't unsettled; a shape that keeps paying a recording's cost is.
    assert profile.summarize_pair([5.0, 5.0, 5.0, 5.0, 5.0, 50.0], [1, 0, 0, 0, 0, 0])["settled"]
    assert not profile.summarize_pair([5.0, 50.0, 5.0, 50.0, 50.0, 5.0], [1, 0, 0, 0, 0, 0])["settled"]
    assert profile.summarize_pair([5.0, 5.0], None)["record_launches"] is None
    # A pair that records twice, or never, breaks "each records one".
    pairs = {
        "a": profile.summarize_pair([1.0, 1.0], [1, 0]),
        "b": profile.summarize_pair([1.0, 1.0], [0, 1]),
    }
    assert profile.launches_recorded_once(pairs) == (True, {"1": 1, "2": 1})
    pairs["c"] = profile.summarize_pair([1.0, 1.0], [1, 1])
    pairs["d"] = profile.summarize_pair([1.0, 1.0], [0, 0])
    assert profile.launches_recorded_once(pairs) == (False, {"1": 1, "2": 1, "1,2": 1, "none": 1})


# ---------------------------------------------------------------------------
# The profile's runs
# ---------------------------------------------------------------------------

def test_a_cpu_smoke_run_writes_its_json_and_skips_cuda_parts_with_notes(tmp_path, capsys) -> None:
    output = tmp_path / "profile.json"
    assert profile.main(["--cpu", "--smoke", "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "Summary" in printed
    report = json.loads(output.read_text())
    assert report["mode"] == "cpu"
    assert set(report["sections"]) == set(profile.ALL_SECTIONS)
    assert "error" not in json.dumps(report["sections"]["model_time"]["rows"])
    assert report["sections"]["compile"]["status"] == "skipped"
    statuses = _statuses(report["checks"])
    assert statuses["CUDA graph record and replay"] == "SKIP"
    assert statuses["memory growth"] == "SKIP"
    assert "FAIL" not in statuses.values()
    # v0's padded and three-tensor paths are the same function as its flat one.
    assert report["sections"]["model_time"]["three_tensors_match_v0"]["logit_max_abs"] < 1e-4
    assert report["pool"]["states"] == 256
    for pair in report["sections"]["launches"]["pairs"].values():
        assert len(pair["ms"]) == 4


class _StubCuda:
    """Just the CUDA surface ``graph_timing`` calls, over the CPU: a graph is the
    forward it captured, re-run with the current inputs."""

    class Stream:
        def wait_stream(self, other: object) -> None:
            pass

    class Event:
        def __init__(self, enable_timing: bool = False) -> None:
            pass

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other: object) -> float:
            return 2.0

    class CUDAGraph:
        def replay(self) -> None:
            pass

    class _Context:
        def __init__(self, *_: object) -> None:
            pass

        def __enter__(self) -> None:
            pass

        def __exit__(self, *_: object) -> None:
            pass

    stream = _Context
    graph = _Context

    @staticmethod
    def current_stream() -> _StubCuda.Stream:
        return _StubCuda.Stream()

    @staticmethod
    def synchronize() -> None:
        pass


def test_the_graph_timing_runs_through_on_a_stubbed_cuda(monkeypatch) -> None:
    """``graph_timing`` can't run here on a real CUDA; this catches its typos and
    its report's shape. The 5090's timing is the only real test of it."""
    config = profile._model_config(
        TrainingConfig.from_json(Path(profile.V0_CONFIG).read_text()), profile.replace(profile.CPU_MODE, d_model=32, num_layers=1),
    )
    game = profile.get_game(config)
    pool = profile.make_pool(game, config, 32)
    mode = profile.CPU_MODE
    num_tokens = game.num_tokens(config.effective_max_players)
    k = profile.launch_k(pool, 2, (0,))
    data = profile.flat_inputs(pool, 2, 0, mode, k=k, num_tokens=num_tokens, edges=game.max_relation_edges)
    torch.manual_seed(0)
    net = profile.create_model(config).eval()
    for name in ("Stream", "Event", "CUDAGraph", "stream", "graph", "current_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, getattr(_StubCuda, name))
    with torch.inference_mode():
        # A stub's replay doesn't rerun, so "changed" inputs are the same ones.
        result = profile.graph_timing(net, data, tuple(t.clone() for t in data), {}, iterations=2, rounds=2)
    assert result["gpu_ms"] == 1.0  # the stub's 2 ms over 2 replays
    assert result["model_states_per_second"] == 2000.0
    assert result["dynamo_graph_breaks"] == {}
    assert result["unique_graphs"] == 1
    assert result["eager_vs_compiled"]["logit_max_abs"] < 1e-3


def test_the_three_tensor_variant_goes_where_its_model_is() -> None:
    """Built in eval mode, on its model's device: a miss here only shows on a GPU."""
    config = profile._model_config(
        TrainingConfig.from_json(Path(profile.V0_CONFIG).read_text()), profile.replace(profile.CPU_MODE, d_model=32, num_layers=1),
    )
    net = profile.create_model(config).train()
    variant = profile.three_tensor_variant(net)
    assert not variant.training
    assert next(variant.parameters()).device == next(net.parameters()).device


def _run_script(script: str, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(Path(script).resolve()), *args],
        capture_output=True, text=True, timeout=120, cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), **(env or {})},
    )


def test_the_profile_fails_with_a_message_outside_the_repo_root(tmp_path) -> None:
    result = _run_script("cloud/profile_1889.py", "--cpu", cwd=tmp_path)
    assert result.returncode != 0
    assert "run from the repo root" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("missing", ["train.eval_server", "g1889.state"])
def test_the_profile_fails_with_a_message_without_a_build(missing: str, tmp_path) -> None:
    code = (
        "import runpy, sys\n"
        f"sys.modules[{missing!r}] = None\n"  # as a missing extension imports
        f"sys.argv = ['cloud/profile_1889.py', '--cpu', '--output', {str(tmp_path / 'out.json')!r}]\n"
        "runpy.run_path('cloud/profile_1889.py', run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "setup.py build_ext --inplace" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_profile_asks_for_a_flag_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parser_args = profile.argparse.Namespace(cpu=False, rocm=False, compile=False, smoke=False)
    with pytest.raises(SystemExit, match="no CUDA GPU visible"):
        profile._mode_from_args(parser_args)
    parser_args.rocm = True
    with pytest.raises(SystemExit, match="--rocm needs"):
        profile._mode_from_args(parser_args)
    parser_args.cpu = True
    with pytest.raises(SystemExit, match="one of --cpu and --rocm"):
        profile._mode_from_args(parser_args)


# ---------------------------------------------------------------------------
# The banner and --help
# ---------------------------------------------------------------------------

def _banner(tmp_path: Path, config_file: str, capsys) -> str:
    config = TrainingConfig.from_json(Path(config_file).read_text())
    config.tensorboard_dir = str(tmp_path)
    logger = TrainingLogger(str(tmp_path))
    logger.console.width = 200
    capsys.readouterr()
    logger.log_training_start(config, device="cuda")
    logger.close()
    return capsys.readouterr().out


def test_a_1889_banner_shows_only_its_models_fields(tmp_path, capsys) -> None:
    banner = _banner(tmp_path, "train_configs/1889-v0.json", capsys)
    for line in ("d_proj", "Fourier", "v2 only", "v3 only", "Phase conditioning", "Relation input mixing"):
        assert line not in banner
    assert "d_model=256, heads=4, layers=15" in banner
    assert "Zero-sum values" in banner
    assert "nn/transformer-1889.py" in banner


def test_an_rss_banner_reads_as_it_did(tmp_path, capsys) -> None:
    banner = _banner(tmp_path, "train_configs/v3-10-01-26.json", capsys)
    for line in (
        "d_model=256, d_proj=64, heads=4", "Phase conditioning (v2 only)", "Relation input mixing (v3 only)",
        "Zero-sum values (v3 only)", "Fourier bands=4",
    ):
        assert line in banner


def test_help_names_no_one_game() -> None:
    assert "Rolling Stock" not in _build_parser().format_help()
    assert "AlphaZero self-play training" in _build_parser().format_help()
