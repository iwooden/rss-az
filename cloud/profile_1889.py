"""The 8.5 profile of the first 1889 run (1889/model-design.md §5.3, §5.7, §6.5).

Standalone; run from the repo root. It imports production code, compiles through
the real ``get_compile_kwargs`` and ``_warmup_compiled_model``, and feeds real
states from random playouts. It writes one JSON file as it goes, so a crash keeps
what it had, and prints a pass/fail summary. Send the JSON back.

    python cloud/profile_1889.py                # the 5090: compiled, bf16
    python cloud/profile_1889.py --cpu          # uncompiled CPU, CUDA parts skipped
    python cloud/profile_1889.py --cpu --compile
    python cloud/profile_1889.py --rocm         # uncompiled ROCm, fp32, small batches

Sections, each failing alone:

- ``model_time``: model-only time at batch 64 and 256 as ``v3-cloud-profile.json``
  measured v3 (Inductor without its graphs, then an explicit CUDA graph, timed by
  CUDA events), v3 against v0, with v0 on the padded records of the bias path it
  replaced and at 8 heads;
- ``compile``: ``torch.compile`` with graph breaks and recompiles counted;
- ``launches``: a serve-loop launch at every (B, K) pair, by launch number, with
  outputs against eager at each shape's first launches and at replay;
- ``replay``: compiled against eager on new inputs and after a weight sync, and the
  same states at different K;
- ``memory``: recorded graphs and GPU memory over many launches;
- ``train_steps``: the probe's training steps a second, at each of its shapes.

The bias variants here are diagnostics, not the model's paths. CUDA graphs' record
and replay is what this torch does, read off the run, not assumed from its source.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, NoReturn


def _fail(message: str) -> NoReturn:
    sys.exit(f"profile_1889: {message}")


REQUIRED_FILES = ("nn/transformer-1889.py", "train_configs/1889-v0.json", "train_configs/v3-10-01-26.json")
for _path in REQUIRED_FILES:
    if not Path(_path).exists():
        _fail(f"no {_path}: run from the repo root, on the az-1889 branch")
sys.path.insert(0, str(Path.cwd()))

try:
    import numpy as np
    import torch

    from games import STATUS_GAME_OVER, STATUS_OK, Game, get_game
    from nn import create_model, get_model_input_spec
    from nn.model_contract import flatten_relation_records
    from train.config import TrainingConfig
    from train.eval_server import (
        SharedEvalBuffers,
        _flat_filler_rows,
        _resolve_flat_launch_rows,
        _warmup_compiled_model,
    )
    from train.gpu import detect_gpu
    from train.replay_buffer import ReplayBuffer
    from train.trainer import Trainer
except Exception as exc:  # ImportError, or a stale extension's ValueError
    _fail(
        f"cannot import the pipeline ({type(exc).__name__}: {exc}). Activate the venv, and build "
        "with: python setup.py build_ext --inplace (after a pull: python setup.py clean_build)"
    )


REVISION = 1
V0_CONFIG = "train_configs/1889-v0.json"
V3_CONFIG = "train_configs/v3-10-01-26.json"
ALL_SECTIONS = ("model_time", "compile", "launches", "replay", "memory", "train_steps")

# v3-cloud-profile.json's v3 at v3-10-01-26's settings (mixing off), for reference
# only: the ratio comes from this run's own v3. Same box, 2026-10.
V3_REFERENCE_MS = {64: 1.843, 256: 4.354}

# Section 5.3 and 6.5's expectations.
RATIO_EXPECTED = (1.7, 2.0)
MIN_PROBE_STEPS_PER_SECOND = 5.0  # 40K steps in about two hours at v0's size
LOGIT_MAX_ABS = 0.05
VALUE_MAX_ABS = 0.01
POLICY_MAX_TV = 0.01
MEMORY_GROWTH_MB = 128
SETTLED_SPREAD = 2.0  # a settled shape's median late launch against its fastest

# The probe's shapes (train/capacity_probe.py).
PROBE_SHAPES = {"256x15": (256, 15, 4), "192x12": (192, 12, 4), "384x18_6h": (384, 18, 6), "256x15_8h": (256, 15, 8)}


@dataclass(frozen=True)
class Mode:
    name: str  # "cuda", "rocm", or "cpu"
    device: str
    compiled: bool  # the serve path goes through torch.compile
    graphs: bool  # CUDA graphs and CUDA events
    autocast: bool  # bf16 under autocast, fp16 tokens; else fp32
    sizes: tuple[int, ...]  # model-only batches
    buckets: tuple[int, ...]  # launch batches
    ks: tuple[int, ...]  # flat rows per state
    launches_per_pair: int
    memory_cycles: int
    iterations: int  # replays per timing round
    rounds: int
    train_batch: int
    train_steps: int
    d_model: int = 256
    num_layers: int = 15
    pool_states: int = 2048
    probe_shapes: tuple[str, ...] = tuple(PROBE_SHAPES)
    output: str = "1889-cloud-profile.json"


CUDA_MODE = Mode(
    "cuda", "cuda", compiled=True, graphs=True, autocast=True,
    sizes=(64, 256), buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256), ks=(256, 512, 1024, 2048, 4096),
    launches_per_pair=12, memory_cycles=200, iterations=100, rounds=5, train_batch=256, train_steps=60,
)
ROCM_MODE = Mode(
    "rocm", "cuda", compiled=False, graphs=False, autocast=False,
    sizes=(16, 64), buckets=(8, 16, 32, 64), ks=(256, 1024), launches_per_pair=4, memory_cycles=3,
    iterations=3, rounds=2, train_batch=64, train_steps=10, probe_shapes=("256x15",),
    output="1889-profile-rocm.json",
)
CPU_MODE = Mode(
    "cpu", "cpu", compiled=False, graphs=False, autocast=False,
    sizes=(4, 16), buckets=(1, 2, 4, 8, 16), ks=(256, 512), launches_per_pair=4, memory_cycles=2,
    iterations=2, rounds=2, train_batch=16, train_steps=4, probe_shapes=("256x15",), d_model=128, num_layers=4,
    output="1889-profile-cpu.json",
)


# ---------------------------------------------------------------------------
# Real states
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    tokens: np.ndarray  # (T, token_dim) fp32
    legal: np.ndarray  # (unified_dim,) bool
    records: np.ndarray  # (E, 4) uint8, zero past the real ones
    record_count: int
    phase: int
    players: int
    row: np.ndarray  # the raw int16 state


def make_pool(game: Game, config: TrainingConfig, count: int, seed: int = 91023) -> list[Entry]:
    """Reachable states from whole random legal games, mixed over the config's
    player counts. Reservoir sampling reaches late game stages without unbounded
    storage. This is not the phase distribution of MCTS traffic."""
    rng = np.random.default_rng(seed)
    spec = get_model_input_spec(config)
    capacity = config.effective_max_players
    actions = np.empty(game.policy.max_action_size, np.uint16)
    lut = game.policy.build_action_lut().numpy()
    counts = list(config.iter_player_counts())
    pool: list[Entry] = []
    seen = 0
    for number in range(4 * len(counts)):
        players = counts[number % len(counts)]
        state = game.new_game(players, capacity, 1000 + number)
        for turn in range(100_000):
            n = game.enumerate_policy_actions(state, actions)
            phase = int(game.decision_phase(state))
            if phase < 0 or n < 1:
                raise RuntimeError(f"invalid generated decision: phase {phase}, {n} actions")
            # Every fourth decision keeps setup inexpensive.
            if turn % 4 == 0:
                seen += 1
                slot = len(pool) if len(pool) < count else int(rng.integers(seen))
                if slot < count:
                    tokens = np.empty((spec.num_tokens, spec.token_dim), np.float32)
                    game.fill_tokens(state, tokens, max_players=capacity, layout_version=spec.layout_version)
                    records = np.zeros((game.max_relation_edges, 4), np.uint8)
                    record_count = int(game.fill_relations_sparse(state, records, max_players=capacity))
                    legal = np.zeros(game.policy.unified_dim, bool)
                    legal[lut[phase, actions[:n]]] = True
                    entry = Entry(tokens, legal, records, record_count, phase, players, state._array.copy())
                    if slot == len(pool):
                        pool.append(entry)
                    else:
                        pool[slot] = entry
            status = game.apply_action(state, int(rng.choice(actions[:n])))
            if status == STATUS_GAME_OVER:
                break
            if status != STATUS_OK:
                raise RuntimeError(f"unexpected driver status {status}")
    if not pool:
        raise RuntimeError("no states generated")
    rng.shuffle(pool)  # type: ignore[arg-type]
    return pool


def _rows_at(pool: list[Entry], batch: int, offset: int) -> list[Entry]:
    return [pool[(i + offset) % len(pool)] for i in range(batch)]


def _wire_tokens(entries: list[Entry], mode: Mode) -> torch.Tensor:
    tokens = torch.from_numpy(np.stack([e.tokens for e in entries]))
    return tokens.to(device=mode.device, dtype=torch.float16 if mode.autocast else torch.float32)


def _legal(entries: list[Entry], mode: Mode) -> torch.Tensor:
    return torch.from_numpy(np.stack([e.legal for e in entries])).to(mode.device)


def flat_inputs(
    pool: list[Entry], batch: int, offset: int, mode: Mode, *, k: int, num_tokens: int, edges: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A bucketed launch's inputs: ``batch × k`` flat rows, real then filler."""
    entries = _rows_at(pool, batch, offset)
    flat = flatten_relation_records(np.stack([e.records for e in entries]))
    rows = batch * k
    filler = _flat_filler_rows(rows - len(flat), edges, num_tokens)
    relations = torch.from_numpy(np.concatenate((flat, filler))).to(mode.device)
    return _wire_tokens(entries, mode), _legal(entries, mode), relations


def padded_inputs(
    pool: list[Entry], batch: int, offset: int, mode: Mode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v0's inputs with every state's E records, padding included: each padded
    record lands on one cell, the contention §5.3 asks about."""
    entries = _rows_at(pool, batch, offset)
    records = torch.from_numpy(np.stack([e.records for e in entries])).to(torch.int16)
    batch_ids = torch.arange(batch, dtype=torch.int16)[:, None, None].expand(-1, records.shape[1], 1)
    flat = torch.cat((batch_ids, records), dim=2).reshape(-1, 5).to(mode.device)
    return _wire_tokens(entries, mode), _legal(entries, mode), flat


def padded_wire_inputs(
    pool: list[Entry], batch: int, offset: int, mode: Mode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v3's eval-server inputs: padded uint8 records."""
    entries = _rows_at(pool, batch, offset)
    records = torch.from_numpy(np.stack([e.records for e in entries])).to(mode.device)
    return _wire_tokens(entries, mode), _legal(entries, mode), records


def launch_k(pool: list[Entry], batch: int, offsets: tuple[int, ...]) -> int:
    """The K both of a timing's batches fit, at their shared launch shape."""
    real = max(sum(e.record_count for e in _rows_at(pool, batch, offset)) for offset in offsets)
    return _resolve_flat_launch_rows(real_rows=real, launch_n=batch, batch_shape_mode="bucketed") // batch


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------

def check_output_contract(output: tuple[torch.Tensor, torch.Tensor], legal: torch.Tensor) -> None:
    policy, values = output
    if policy.shape != legal.shape:
        raise RuntimeError(f"policy shape {tuple(policy.shape)} against mask {tuple(legal.shape)}")
    if not (torch.isfinite(policy).all() and torch.isfinite(values).all()):
        raise RuntimeError("non-finite output")
    if not (policy[~legal] == -1e9).all():
        raise RuntimeError("the illegal-slot mask changed")


def output_differences(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    legal: torch.Tensor,
) -> dict[str, float]:
    """Two runs' differences on real rows, with v3-cloud-profile.json's tolerance
    (0.003 + 3%) reported beside the absolute numbers."""
    check_output_contract(actual, legal)
    check_output_contract(expected, legal)
    result: dict[str, float] = {}
    for name, a, b in (
        ("logit", actual[0][legal].float(), expected[0][legal].float()),
        ("value", actual[1].float(), expected[1].float()),
    ):
        delta = (a - b).abs()
        result[f"{name}_count"] = a.numel()
        result[f"{name}_max_abs"] = delta.max().item()
        result[f"{name}_rms"] = delta.square().mean().sqrt().item()
        result[f"{name}_outside_tolerance"] = (delta > 0.003 + 0.03 * b.abs()).sum().item()
    probability = (actual[0].float().softmax(-1) - expected[0].float().softmax(-1)).abs()
    result["policy_max_tv"] = (0.5 * probability.sum(-1)).max().item()
    return result


def merge_differences(differences: list[dict[str, float]]) -> dict[str, float]:
    """Worst of each maximum, and sums of the counts."""
    merged: dict[str, float] = {}
    for difference in differences:
        for key, value in difference.items():
            if key.endswith(("count", "tolerance")):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = max(merged.get(key, 0.0), value)
    return merged


def differences_within_limits(difference: dict[str, float]) -> bool:
    return (
        difference["logit_max_abs"] <= LOGIT_MAX_ABS
        and difference["value_max_abs"] <= VALUE_MAX_ABS
        and difference["policy_max_tv"] <= POLICY_MAX_TV
    )


# ---------------------------------------------------------------------------
# Model-only time
# ---------------------------------------------------------------------------

def three_tensor_variant(base: torch.nn.Module) -> torch.nn.Module:
    """v0 on v3's bias path, a diagnostic: zeros with the records scattered in,
    plus the static bias, plus visibility, three dense tensors a layer (§5.3)."""
    cls: Any = type(base)
    num_dynamic = base._relation_scales.shape[0]  # type: ignore[union-attr]

    class ThreeTensors(cls):
        def _attention_bias(self, relations: Any, hidden_keys: torch.Tensor, layer_idx: int, ref: torch.Tensor) -> torch.Tensor:
            mult = self.relation_bias_mult[layer_idx].to(device=ref.device, dtype=ref.dtype)
            size = self.cfg.num_tokens
            bias = ref.new_zeros(ref.shape[0], self.cfg.num_heads, size, size)
            values = mult.index_select(1, relations.relation_ids) * relations.weights.to(device=ref.device, dtype=ref.dtype)
            bias.reshape(-1).scatter_add_(0, relations.flat_indices, values.reshape(-1))
            static = torch.einsum(
                "rij,hr->hij", self._static_relations.to(device=ref.device, dtype=ref.dtype), mult[:, num_dynamic:],
            )
            bias = bias + static
            visibility = torch.zeros_like(hidden_keys, dtype=bias.dtype).masked_fill(
                hidden_keys, torch.finfo(bias.dtype).min,
            )
            return bias + visibility

    net: torch.nn.Module = ThreeTensors(base.cfg)  # type: ignore[union-attr]
    net.load_state_dict(base.state_dict())
    return net.to(next(base.parameters()).device).eval()


def graph_timing(
    net: torch.nn.Module,
    data: tuple[torch.Tensor, ...],
    changed: tuple[torch.Tensor, ...],
    options: dict[str, Any],
    iterations: int,
    rounds: int,
) -> dict[str, Any]:
    """v3-cloud-profile.json's method: Inductor without its graphs, three warm
    calls on one stream, an explicit graph, a replay on changed inputs, then
    replays timed by CUDA events."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    start = time.perf_counter()
    compiled: Any = torch.compile(net, fullgraph=True, dynamic=False, options=options)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(*data)
        compiled(*data)
        output = compiled(*data)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - start
    eager_comparison = output_differences(output, net(*data), data[1])
    del output
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(*data)
    # Fresh inputs and phase distributions go through the same graph.
    for target, source in zip(data, changed):
        target.copy_(source)
    graph.replay()
    check_output_contract(output, data[1])
    replay_comparison = output_differences(output, compiled(*data), data[1])
    for _ in range(10):
        graph.replay()
    samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        wall_samples.append(1000 * (time.perf_counter() - wall_start) / iterations)
        samples.append(begin.elapsed_time(end) / iterations)
    median = statistics.median(samples)
    result = {
        "gpu_ms": median,
        "gpu_ms_samples": samples,
        "wall_ms": statistics.median(wall_samples),
        "compile_seconds": compile_seconds,
        "model_states_per_second": data[0].shape[0] * 1000 / median,
        "eager_vs_compiled": eager_comparison,
        "graph_replay_vs_compiled": replay_comparison,
        "dynamo_graph_breaks": dict(counters["graph_break"]),
        "unique_graphs": counters["stats"]["unique_graphs"],
    }
    del output, graph, compiled
    return result


def eager_timing(
    net: torch.nn.Module, data: tuple[torch.Tensor, ...], mode: Mode, iterations: int, rounds: int,
) -> dict[str, Any]:
    """Wall time of an uncompiled forward: a harness check, not the 5090's test."""
    check_output_contract(net(*data), data[1])
    samples = []
    for _ in range(rounds):
        begin = time.perf_counter()
        for _ in range(iterations):
            net(*data)
        if mode.device == "cuda":
            torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - begin) / iterations)
    median = statistics.median(samples)
    return {"wall_ms": median, "gpu_ms": median, "model_states_per_second": data[0].shape[0] * 1000 / median}


def _graph_options() -> dict[str, Any]:
    options = dict(detect_gpu("cuda").get_compile_kwargs(for_training=False, eval_batch_shape_mode="bucketed")["options"])
    # Explicit capture is the check here, and avoids nesting automatic graph trees.
    options["triton.cudagraphs"] = False
    return options


def _model_config(base: TrainingConfig, mode: Mode, **fields_: Any) -> TrainingConfig:
    return replace(base, d_model=mode.d_model, num_layers=mode.num_layers, **fields_)


def section_model_time(ctx: Context) -> dict[str, Any]:
    mode = ctx.mode
    v3_config = _model_config(ctx.v3_config, mode)
    v3_pool = make_pool(ctx.v3_game, v3_config, mode.pool_states)
    pool = ctx.pool
    num_tokens = ctx.game.num_tokens(ctx.config.effective_max_players)
    edges = ctx.game.max_relation_edges
    options = _graph_options() if mode.graphs else {}
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "method": (
            "v3-cloud-profile.json's: Inductor, explicit CUDA graph, CUDA events" if mode.graphs
            else "uncompiled wall time: a harness check, not the 5090's test"
        ),
        "v3_reference_ms": V3_REFERENCE_MS,
        "rows": rows,
    }

    def new_v3() -> torch.nn.Module:
        torch.manual_seed(42)
        return create_model(v3_config).to(mode.device).eval()

    def new_v0(num_heads: int = 4) -> torch.nn.Module:
        torch.manual_seed(42)
        return create_model(replace(ctx.config, num_heads=num_heads)).to(mode.device).eval()

    cases: list[tuple[str, Callable[[], torch.nn.Module], str]] = [
        ("v3", new_v3, "wire"),
        ("v0", new_v0, "flat"),
        ("v0_padded_one_tensor", new_v0, "padded"),
        ("v0_padded_three_tensors", lambda: three_tensor_variant(new_v0()), "padded"),
        ("v0_8heads", lambda: new_v0(8), "flat"),
        ("v0_8heads_padded_three_tensors", lambda: three_tensor_variant(new_v0(8)), "padded"),
    ]
    with torch.inference_mode(), _autocast(mode):
        # The diagnostic must compute what the model does.
        batch = mode.sizes[0]
        k = launch_k(pool, batch, (0,))
        flat = flat_inputs(pool, batch, 0, mode, k=k, num_tokens=num_tokens, edges=edges)
        padded = padded_inputs(pool, batch, 0, mode)
        reference = new_v0()(*flat)
        result["padded_matches_v0"] = output_differences(reference, new_v0()(*padded), flat[1])
        result["three_tensors_match_v0"] = output_differences(
            reference, three_tensor_variant(new_v0())(*padded), flat[1],
        )
        del flat, padded, reference
        _release(mode)
        for size in mode.sizes:
            for name, factory, inputs in cases:
                row: dict[str, Any] = {"case": name, "batch": size}
                print(f"  {name}, batch {size}...", flush=True)
                net = None
                try:
                    net = factory()
                    if inputs == "wire":
                        data = padded_wire_inputs(v3_pool, size, 0, mode)
                        changed = padded_wire_inputs(v3_pool, size, size, mode)
                    elif inputs == "padded":
                        data = padded_inputs(pool, size, 0, mode)
                        changed = padded_inputs(pool, size, size, mode)
                    else:
                        k = launch_k(pool, size, (0, size))
                        row["k"] = k
                        data = flat_inputs(pool, size, 0, mode, k=k, num_tokens=num_tokens, edges=edges)
                        changed = flat_inputs(pool, size, size, mode, k=k, num_tokens=num_tokens, edges=edges)
                    row["parameters"] = sum(p.numel() for p in net.parameters())
                    row.update(
                        graph_timing(net, data, changed, options, mode.iterations, mode.rounds) if mode.graphs
                        else eager_timing(net, data, mode, mode.iterations, mode.rounds)
                    )
                    print(f"    {row['gpu_ms']:.3f} ms, {row['model_states_per_second']:,.0f} states/s", flush=True)
                except Exception:
                    row["error"] = traceback.format_exc()
                    print(f"    failed: {_last_line(row['error'])}", flush=True)
                rows.append(row)
                ctx.save()
                del net
                _release(mode)
    result["ratios"] = model_time_ratios(rows, mode.sizes)
    return result


def model_time_ratios(rows: list[dict[str, Any]], sizes: tuple[int, ...]) -> dict[str, dict[str, float]]:
    """Each batch's times against v3, and the bias path's and 8 heads' against v0."""
    ms = {(r["case"], r["batch"]): r["gpu_ms"] for r in rows if "gpu_ms" in r}
    ratios: dict[str, dict[str, float]] = {}
    for size in sizes:
        by_batch: dict[str, float] = {}
        for name, over, under in (
            ("v0_over_v3", "v0", "v3"),
            ("padded_one_tensor_over_v0", "v0_padded_one_tensor", "v0"),
            ("padded_three_tensors_over_v0", "v0_padded_three_tensors", "v0"),
            ("8heads_over_v0", "v0_8heads", "v0"),
            ("8heads_padded_over_padded", "v0_8heads_padded_three_tensors", "v0_padded_three_tensors"),
        ):
            if (over, size) in ms and (under, size) in ms:
                by_batch[name] = ms[(over, size)] / ms[(under, size)]
        ratios[str(size)] = by_batch
    return ratios


# ---------------------------------------------------------------------------
# The serve path
# ---------------------------------------------------------------------------

class CompileEvents:
    """Dynamo's compiles and their guard failures, by counting and by its log.
    Inductor runs the same callbacks for each CUDA graph it records, under another
    trigger; ``CaptureCounter`` counts those."""

    def __init__(self) -> None:
        self.compiles = 0
        self.recompile_reasons: list[str] = []
        self.notes: list[str] = []
        try:
            torch._dynamo.callback.on_compile_end(self._on_compile_end)  # type: ignore[attr-defined]
        except Exception as exc:
            self.notes.append(f"no compile callback: {type(exc).__name__}: {exc}")
        try:
            events = self

            class Handler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    message = record.getMessage()
                    if "guard failure" in message or " - " in message or "Recompiling" in message:
                        events.recompile_reasons.append(message.strip())

            torch._logging.set_logs(recompiles=True)  # type: ignore[attr-defined]
            logging.getLogger("torch._dynamo.guards.__recompiles").addHandler(Handler())
        except Exception as exc:
            self.notes.append(f"no recompile log: {type(exc).__name__}: {exc}")

    def _on_compile_end(self, args: Any) -> None:
        if args.callback_trigger == torch._dynamo.callback.CallbackTrigger.DYNAMO:  # type: ignore[attr-defined]
            self.compiles += 1


class CaptureCounter:
    """CUDA graph recordings as they start, by wrapping ``torch.cuda.graph``."""

    def __init__(self) -> None:
        self.count = 0
        self.available = False
        self.note = ""
        try:
            original = torch.cuda.graph.__enter__

            def enter(graph: Any) -> Any:
                self.count += 1
                return original(graph)

            torch.cuda.graph.__enter__ = enter  # type: ignore[method-assign]
            self.available = True
        except Exception as exc:
            self.note = f"{type(exc).__name__}: {exc}"


def graph_tree_stats() -> dict[str, Any]:
    """Inductor's CUDA graph trees, by private API that a torch may have moved."""
    try:
        from torch._dynamo.utils import counters
        from torch._inductor import cudagraph_trees as trees

        manager = trees.get_manager(torch.cuda.current_device(), create_if_none_exists=False)
        stats: dict[str, Any] = {
            "counters": {k: v for k, v in counters["inductor"].items() if "cudagraph" in k},
        }
        if manager is None:
            return {**stats, "functions": 0, "recorded_nodes": 0}
        stack = list(manager.get_roots())
        roots = len(stack)
        nodes = 0
        while stack:
            node = stack.pop()
            nodes += 1
            stack.extend(child for children in node.children.values() for child in children)
        return {
            **stats,
            "functions": len(manager.ids_to_funcs),
            "warmed_up": len(manager.warmed_up_functions),
            "roots": roots,
            "recorded_nodes": nodes,
            "failed_matches": manager.debug_fail_counter,
        }
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def memory_stats(mode: Mode) -> dict[str, float]:
    if mode.device != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 2**20,
        "reserved_mb": torch.cuda.memory_reserved() / 2**20,
        "device_free_mb": free / 2**20,
        "device_total_mb": total / 2**20,
    }


def nvidia_smi_memory() -> dict[str, Any]:
    """This box's per-process GPU memory, outside torch's view, if the tool is there."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        return {"used_mb_by_pid": [line.strip() for line in out.splitlines() if line.strip()]}
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


class Launcher:
    """The serve loop's launch (``_infer_and_scatter``), in its buffers and order,
    without the workers: gather to pinned host buffers, flatten and round up to the
    launch shape, copy, run under the persistent autocast, softmax, copy back."""

    def __init__(self, game: Game, num_tokens: int, token_dim: int, num_players: int, max_batch: int, mode: Mode) -> None:
        self.mode = mode
        device = mode.device
        pin = device == "cuda"
        edges = game.max_relation_edges
        u_dim = game.policy.unified_dim
        self.pin_s = torch.empty(max_batch, num_tokens, token_dim, dtype=torch.float16, pin_memory=pin)
        self.gpu_s = torch.empty(
            max_batch, num_tokens, token_dim,
            dtype=torch.float16 if mode.autocast else torch.float32, device=device,
        )
        flat_rows = max_batch * edges
        self.pin_rel = torch.empty(flat_rows, 5, dtype=torch.int16, pin_memory=pin)
        self.pin_rel_np = self.pin_rel.numpy()
        self.gpu_rel = torch.empty(flat_rows, 5, dtype=torch.int16, device=device)
        self.filler = _flat_filler_rows(flat_rows, edges, num_tokens)
        self.pin_mask = torch.empty(max_batch, u_dim, dtype=torch.uint8, pin_memory=pin)
        self.gpu_mask = torch.zeros(max_batch, u_dim, dtype=torch.uint8, device=device)
        self.gpu_mask_bool = torch.zeros(max_batch, u_dim, dtype=torch.bool, device=device)
        self.pin_priors = torch.empty(max_batch, u_dim, dtype=torch.float32, pin_memory=pin)
        self.gpu_priors = torch.empty(max_batch, u_dim, dtype=torch.float32, device=device)
        self.pin_val = torch.empty(max_batch, num_players, dtype=torch.float32, pin_memory=pin)
        self.gpu_val = torch.empty(max_batch, num_players, dtype=torch.float32, device=device)

    @staticmethod
    def wire(entries: list[Entry]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """A launch's shared-memory rows, before the serve loop gathers them. A
        server gets these from its workers, so a launch's time leaves them out."""
        return (
            torch.from_numpy(np.stack([e.tokens for e in entries]).astype(np.float16)),
            torch.from_numpy(np.stack([e.legal for e in entries]).astype(np.uint8)),
            np.stack([e.records for e in entries]),
        )

    def inputs(self, wire: tuple[torch.Tensor, torch.Tensor, np.ndarray], launch_n: int, k: int) -> int:
        """Gather a launch into the pinned buffers and start its copies; returns its
        relation rows."""
        tokens, masks, records = wire
        actual_n = len(records)
        self.pin_s[:actual_n] = tokens
        self.pin_mask[:actual_n] = masks
        flat = flatten_relation_records(records)
        real_rows = len(flat)
        relation_rows = launch_n * k
        if real_rows > relation_rows:
            raise RuntimeError(f"{real_rows} real rows don't fit {launch_n} x {k}")
        self.pin_rel_np[:real_rows] = flat
        self.pin_rel_np[real_rows:relation_rows] = self.filler[:relation_rows - real_rows]
        batch = self.gpu_s[:launch_n]
        batch[:actual_n].copy_(self.pin_s[:actual_n], non_blocking=True)
        self.gpu_rel[:relation_rows].copy_(self.pin_rel[:relation_rows], non_blocking=True)
        self.gpu_mask[:actual_n].copy_(self.pin_mask[:actual_n], non_blocking=True)
        if launch_n > actual_n:
            batch[actual_n:launch_n].zero_()
            self.gpu_mask[actual_n:launch_n].zero_()
            self.gpu_mask_bool[actual_n:launch_n].zero_()
        self.gpu_mask_bool[:launch_n].copy_(self.gpu_mask[:launch_n], non_blocking=True)
        return relation_rows

    def run(
        self, model: Any, launch_n: int, actual_n: int, relation_rows: int, *, keep: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """One forward and its copy back, with its outputs dead on return. ``keep``
        returns copies of the real rows' logits, values, and mask."""
        kept = None
        states = self.gpu_s[:launch_n]
        mask = self.gpu_mask_bool[:launch_n]
        with torch.inference_mode():
            logits, values = model(states, mask, self.gpu_rel[:relation_rows])
            self.gpu_priors[:actual_n] = logits[:actual_n].softmax(dim=1).to(torch.float32)
            self.gpu_val[:actual_n] = values[:actual_n].to(torch.float32)
            self.pin_priors[:actual_n].copy_(self.gpu_priors[:actual_n], non_blocking=True)
            self.pin_val[:actual_n].copy_(self.gpu_val[:actual_n], non_blocking=True)
            if keep:
                kept = (logits[:actual_n].clone(), values[:actual_n].clone(), self.gpu_mask_bool[:actual_n].clone())
        if self.mode.device == "cuda":
            torch.cuda.synchronize()
        return kept

    def launch(
        self, model: Any, entries: list[Entry], launch_n: int, k: int, *, keep: bool = False,
    ) -> tuple[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        """A launch and its time in ms, and ``run``'s copies if ``keep``."""
        wire = self.wire(entries)
        begin = time.perf_counter()
        relation_rows = self.inputs(wire, launch_n, k)
        kept = self.run(model, launch_n, len(entries), relation_rows, keep=keep)
        return 1000 * (time.perf_counter() - begin), kept


def choose_states(by_k: dict[int, list[Entry]], launch_n: int, k: int, number: int, rng: np.random.Generator) -> list[Entry]:
    """Real states for a launch at (B, K), in the serve loop's two kinds of bucket
    fill: full on even launch numbers, just past half on odd ones."""
    actual_n = launch_n if number % 2 == 0 or launch_n == 1 else launch_n // 2 + 1
    pool = by_k[k]
    return [pool[i] for i in rng.integers(0, len(pool), actual_n)]


def summarize_pair(ms: list[float], captures: list[int] | None) -> dict[str, Any]:
    """One (B, K)'s launches by number: when it recorded, and what it settled to."""
    late = ms[len(ms) // 2:]
    settled = statistics.median(late)
    return {
        "ms": [round(x, 3) for x in ms],
        "record_launches": None if captures is None else [i + 1 for i, c in enumerate(captures) if c],
        "settled_ms": settled,
        "settled": settled <= SETTLED_SPREAD * min(late),
    }


def launches_recorded_once(pairs: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, int]]:
    """Whether every pair recorded exactly once, and the launch numbers they did it at."""
    histogram: dict[str, int] = {}
    once = True
    for summary in pairs.values():
        recorded = summary["record_launches"]
        if recorded is None or len(recorded) != 1:
            once = False
        key = "none" if not recorded else ",".join(str(n) for n in recorded)
        histogram[key] = histogram.get(key, 0) + 1
    return once, histogram


def section_compile(ctx: Context) -> dict[str, Any]:
    """Compile and warm up as ``_eval_server_serve`` does."""
    mode = ctx.mode
    if not mode.compiled:
        return {"status": "skipped", "note": "uncompiled mode: pass --cpu --compile, or run on the 5090"}
    from torch._dynamo.utils import counters

    if mode.graphs:
        compile_kwargs = detect_gpu("cuda").get_compile_kwargs(for_training=False, eval_batch_shape_mode="bucketed")
    else:
        # The real function, whatever the device: --cpu --compile checks this path.
        from train.gpu.nvidia import get_compile_kwargs

        compile_kwargs = get_compile_kwargs(for_training=False, eval_batch_shape_mode="bucketed")
    spec = get_model_input_spec(ctx.config)
    shared = SharedEvalBuffers(
        ctx.game, num_workers=1, batch_size=ctx.config.search_batch_size,
        num_players=ctx.config.effective_max_players, input_spec=spec,
    )
    torch._dynamo.reset()
    counters.clear()
    torch.manual_seed(42)
    base = create_model(ctx.config).to(mode.device).eval()
    begin = time.perf_counter()
    events = CompileEvents()
    model: Any = torch.compile(base, **compile_kwargs)
    model.eval()
    _warmup_compiled_model(
        model, shared, torch.device(mode.device),
        batch_shape_mode="bucketed",
        max_launch_batch=max(mode.buckets),
        autocast_dtype=torch.bfloat16 if mode.autocast else None,
        input_dtype=torch.float16 if mode.autocast else torch.float32,
    )
    if mode.device == "cuda":
        torch.cuda.synchronize()
    ctx.serving = Serving(base, model, events, CaptureCounter() if mode.graphs else None)
    return {
        "kwargs": repr(compile_kwargs),
        "warmup_seconds": time.perf_counter() - begin,
        "graph_breaks": dict(counters["graph_break"]),
        "unique_graphs": counters["stats"]["unique_graphs"],
        "calls_captured": counters["stats"]["calls_captured"],
        "compiles": events.compiles,
        "recompile_log": list(events.recompile_reasons),
        "notes": events.notes,
        "memory_after_warmup": memory_stats(mode),
    }


def _serving_model(ctx: Context) -> tuple[Any, Any, CompileEvents | None, CaptureCounter | None]:
    """The compiled model and its counters, or the uncompiled model."""
    if ctx.serving is not None:
        return ctx.serving.base, ctx.serving.model, ctx.serving.events, ctx.serving.captures
    torch.manual_seed(42)
    base = create_model(ctx.config).to(ctx.mode.device).eval()
    ctx.serving = Serving(base, base, None, None)
    return base, base, None, None


def section_launches(ctx: Context) -> dict[str, Any]:
    mode = ctx.mode
    base, model, events, captures = _serving_model(ctx)
    rng = np.random.default_rng(5)
    with _autocast(mode):
        launcher = ctx.launcher()
        pairs: dict[str, dict[str, Any]] = {}
        comparisons: dict[str, list[dict[str, float]]] = {}
        compiles_before = events.compiles if events else 0
        reasons_before = len(events.recompile_reasons) if events else 0
        for launch_n in mode.buckets:
            for k in mode.ks:
                key = f"B={launch_n},K={k}"
                ms: list[float] = []
                recorded: list[int] = []
                differences = []
                for number in range(mode.launches_per_pair):
                    entries = choose_states(ctx.by_k, launch_n, k, number, rng)
                    before = captures.count if captures else 0
                    # Each shape's run without a graph, its recording, and its
                    # first replay, against eager on their own inputs.
                    elapsed, kept = launcher.launch(model, entries, launch_n, k, keep=number < 3)
                    ms.append(elapsed)
                    recorded.append((captures.count - before) if captures else 0)
                    if kept is not None:
                        eager = launcher.run(base, launch_n, len(entries), launch_n * k, keep=True)
                        assert eager is not None
                        differences.append(output_differences(kept[:2], eager[:2], kept[2]))
                pairs[key] = summarize_pair(ms, recorded if captures and captures.available else None)
                comparisons[key] = differences
            ctx.save()
        result: dict[str, Any] = {
            "pairs": pairs,
            "against_eager_by_launch": {
                key: [{k: v for k, v in d.items() if not k.endswith("count")} for d in differences]
                for key, differences in comparisons.items()
            },
            "against_eager": merge_differences([d for ds in comparisons.values() for d in ds]),
            "graph_trees": graph_tree_stats() if mode.graphs else {"note": "no CUDA graphs in this mode"},
        }
        if events:
            result["compiles_after_warmup"] = events.compiles - compiles_before
            result["recompile_log_after_warmup"] = events.recompile_reasons[reasons_before:]
        if captures and captures.available:
            once, histogram = launches_recorded_once(pairs)
            result["captures"] = captures.count
            result["recorded_once_each"] = once
            result["record_launch_histogram"] = histogram
        elif captures:
            result["captures"] = f"not counted: {captures.note}"
        return result


def _against_eager(
    launcher: Launcher, model: Any, base: Any, entries: list[Entry], launch_n: int, k: int,
) -> dict[str, float]:
    """The same launch's replayed and eager outputs on its real rows."""
    relation_rows = launcher.inputs(launcher.wire(entries), launch_n, k)
    kept = launcher.run(model, launch_n, len(entries), relation_rows, keep=True)
    assert kept is not None
    eager = launcher.run(base, launch_n, len(entries), relation_rows, keep=True)
    assert eager is not None
    return output_differences(kept[:2], eager[:2], kept[2])


def section_replay(ctx: Context) -> dict[str, Any]:
    """Compiled against eager once shapes replay: new inputs at every pair, after a
    weight sync, and the same states at different K."""
    mode = ctx.mode
    base, model, _, _ = _serving_model(ctx)
    rng = np.random.default_rng(6)
    result: dict[str, Any] = {}
    with _autocast(mode):
        launcher = ctx.launcher()
        by_pair: dict[str, dict[str, float]] = {}
        for launch_n in mode.buckets:
            for k in mode.ks:
                entries = choose_states(ctx.by_k, launch_n, k, 0, rng)
                # A replay's new inputs, on their first and second forward.
                by_pair[f"B={launch_n},K={k}"] = merge_differences([
                    _against_eager(launcher, model, base, entries, launch_n, k) for _ in range(2)
                ])
        result["by_pair"] = by_pair
        result["against_eager"] = merge_differences(list(by_pair.values()))

        # K changes nothing but the filler: the same states at each K, compiled.
        launch_n = next((b for b in reversed(mode.buckets) if b <= 64), mode.buckets[-1])
        entries = choose_states(ctx.by_k, launch_n, mode.ks[0], 0, rng)
        outputs = []
        wire = launcher.wire(entries)
        for k in mode.ks:
            relation_rows = launcher.inputs(wire, launch_n, k)
            kept = launcher.run(model, launch_n, launch_n, relation_rows, keep=True)
            assert kept is not None
            outputs.append(kept)
        result["filler_invariance"] = merge_differences([
            output_differences(other[:2], outputs[0][:2], outputs[0][2]) for other in outputs[1:]
        ])

        # A weight sync: load_state_dict in place, as _check_weight_sync does. The
        # replay must follow, or self-play would run on stale weights.
        saved = {name: value.clone() for name, value in base.state_dict().items()}
        relation_rows = launcher.inputs(wire, launch_n, mode.ks[0])
        before = launcher.run(model, launch_n, launch_n, relation_rows, keep=True)
        assert before is not None
        torch.manual_seed(7)
        base.load_state_dict({
            name: value + 0.02 * torch.randn_like(value) if value.is_floating_point() else value
            for name, value in saved.items()
        }, strict=True)
        if mode.device == "cuda":
            torch.cuda.synchronize()
        after = launcher.run(model, launch_n, launch_n, relation_rows, keep=True)
        eager = launcher.run(base, launch_n, launch_n, relation_rows, keep=True)
        assert after is not None and eager is not None
        result["weight_sync"] = {
            "replay_against_eager": output_differences(after[:2], eager[:2], after[2]),
            "replay_against_before": output_differences(after[:2], before[:2], after[2]),
        }
        base.load_state_dict(saved, strict=True)
    return result


def section_memory(ctx: Context) -> dict[str, Any]:
    mode = ctx.mode
    _, model, events, captures = _serving_model(ctx)
    rng = np.random.default_rng(8)
    samples: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {}
    launches = 0
    with _autocast(mode):
        launcher = ctx.launcher()
        sample_every = max(1, mode.memory_cycles // 20)
        for cycle in range(mode.memory_cycles):
            for launch_n in mode.buckets:
                for k in mode.ks:
                    entries = choose_states(ctx.by_k, launch_n, k, cycle, rng)
                    latencies.setdefault(f"B={launch_n},K={k}", []).append(
                        launcher.launch(model, entries, launch_n, k)[0]
                    )
                    launches += 1
            if cycle % sample_every == 0 or cycle == mode.memory_cycles - 1:
                samples.append({
                    "cycle": cycle,
                    "launches": launches,
                    "captures": captures.count if captures else None,
                    "compiles": events.compiles if events else None,
                    **memory_stats(mode),
                })
                ctx.save()
    first, last = samples[0], samples[-1]
    return {
        "launches": launches,
        "pairs": len(latencies),
        "samples": samples,
        "growth_mb": {
            name: last[name] - first[name]
            for name in ("allocated_mb", "reserved_mb", "device_free_mb") if name in first
        },
        # After the first cycle, which records any shape not yet recorded.
        "new_captures": (last["captures"] - first["captures"]) if captures else None,
        "new_compiles": (last["compiles"] - first["compiles"]) if events else None,
        "graph_trees": graph_tree_stats() if mode.graphs else {"note": "no CUDA graphs in this mode"},
        "nvidia_smi": nvidia_smi_memory() if mode.graphs else {"note": "not the 5090"},
        "median_ms": {key: statistics.median(ms) for key, ms in latencies.items()},
        "p99_ms": {key: float(np.percentile(ms, 99)) for key, ms in latencies.items()},
    }


def section_train_steps(ctx: Context) -> dict[str, Any]:
    """The real trainer on a buffer of playout rows, fp32 and uncompiled, as the
    probe trains. Targets are uniform over legal slots: timing, not learning."""
    mode = ctx.mode
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(f"torch {torch.__version__} has no optimizer Muon; install.sh pins the torch to use")
    ctx.release_serving()
    config = ctx.config
    game = ctx.game
    max_players = config.effective_max_players
    buffer = ReplayBuffer(
        game, len(ctx.pool), game.state_size(max_players), max_players,
        min_players=config.effective_min_players, max_players=max_players,
        priority_fraction=config.replay_priority_fraction, priority_exponent=config.replay_priority_exponent,
    )
    legal = np.stack([e.legal for e in ctx.pool]).astype(np.uint8)
    buffer.add_stacked(
        states=np.stack([e.row for e in ctx.pool]),
        phase_ids=np.array([e.phase for e in ctx.pool], dtype=np.int8),
        legal_masks=legal,
        policy_targets=(legal / legal.sum(axis=1, keepdims=True)).astype(np.float32),
        value_targets=np.zeros((len(ctx.pool), max_players), dtype=np.float32),
        player_counts=np.array([e.players for e in ctx.pool], dtype=np.uint8),
        priorities=np.ones(len(ctx.pool), dtype=np.float32),
    )
    results: dict[str, Any] = {"batch": mode.train_batch, "steps": mode.train_steps, "shapes": {}}
    for name in mode.probe_shapes:
        print(f"  {name}...", flush=True)
        d_model, num_layers, num_heads = PROBE_SHAPES[name]
        if mode.d_model != 256:  # a small mode's shape, for the script's check
            d_model, num_layers = mode.d_model, mode.num_layers
        shape_config = replace(
            config, d_model=d_model, num_layers=num_layers, num_heads=num_heads, batch_size=mode.train_batch,
            min_buffer_size=max(config.min_buffer_size, mode.train_batch),
            replay_ratio=0.0,  # steps here have no epochs: the step LR schedule
        )
        torch.manual_seed(42)
        model = create_model(shape_config).to(mode.device)
        trainer = Trainer(model, shape_config, torch.device(mode.device))
        rng = np.random.default_rng(42)
        model.train()
        # The first steps build scratch and kernels.
        for _ in range(min(5, mode.train_steps)):
            trainer.train_step(buffer, mode.train_batch, rng)
        if mode.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        for _ in range(mode.train_steps):
            trainer.train_step(buffer, mode.train_batch, rng)
        seconds = time.perf_counter() - begin
        rate = mode.train_steps / seconds
        results["shapes"][name] = {
            "steps_per_second": rate,
            "hours_for_40k_steps": 40_000 / rate / 3600,
            "parameters": sum(p.numel() for p in model.parameters()),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if mode.device == "cuda" else None,
        }
        print(f"    {rate:.1f} steps/s", flush=True)
        del trainer, model
        _release(mode)
        ctx.save()
    return results


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass
class Serving:
    base: Any
    model: Any
    events: CompileEvents | None
    captures: CaptureCounter | None


@dataclass
class Context:
    mode: Mode
    config: TrainingConfig
    v3_config: TrainingConfig
    game: Game
    v3_game: Game
    pool: list[Entry]
    by_k: dict[int, list[Entry]]
    save: Callable[[], None]
    serving: Serving | None = None
    _launcher: Launcher | None = field(default=None, repr=False)

    def launcher(self) -> Launcher:
        """One set of serve-loop buffers: their addresses are what a CUDA graph records."""
        if self._launcher is None:
            max_players = self.config.effective_max_players
            spec = get_model_input_spec(self.config)
            self._launcher = Launcher(
                self.game, spec.num_tokens, spec.token_dim, max_players, max(self.mode.buckets), self.mode,
            )
        return self._launcher

    def release_serving(self) -> None:
        self.serving = None
        self._launcher = None
        if self.mode.compiled:
            torch._dynamo.reset()
        if self.mode.graphs:
            try:
                from torch._inductor.cudagraph_trees import reset_cudagraph_trees

                reset_cudagraph_trees()
            except Exception as exc:
                print(f"  (cudagraph reset: {type(exc).__name__}: {exc})", flush=True)
        _release(self.mode)


def _release(mode: Mode) -> None:
    gc.collect()
    if mode.device == "cuda":
        torch.cuda.empty_cache()


def _autocast(mode: Mode) -> Any:
    """The serve loop's persistent context."""
    return torch.autocast(
        "cuda" if mode.device == "cuda" else "cpu",
        dtype=torch.bfloat16,
        enabled=mode.autocast,
        cache_enabled=False,
    )


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _mode_from_args(args: argparse.Namespace) -> Mode:
    if args.cpu and args.rocm:
        _fail("pass one of --cpu and --rocm")
    if args.cpu:
        mode = replace(CPU_MODE, compiled=args.compile)
    elif args.rocm:
        if args.compile:
            _fail("--rocm stays uncompiled: compiled checks on ROCm test ROCm")
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
            _fail("--rocm needs a ROCm torch with a visible GPU")
        mode = ROCM_MODE
    else:
        if not torch.cuda.is_available():
            _fail(
                "no CUDA GPU visible. On the 5090, check nvidia-smi and that the venv's torch is a CUDA "
                "build; a sandbox can hide the GPU. To check the script alone, pass --cpu or --rocm."
            )
        if getattr(torch.version, "hip", None) is not None:
            _fail("this torch is a ROCm build; pass --rocm")
        mode = CUDA_MODE
    if args.smoke:
        mode = replace(
            mode, d_model=32, num_layers=1, pool_states=256, launches_per_pair=4, memory_cycles=2, iterations=2,
            rounds=2, train_steps=2, sizes=mode.sizes[:1],
        )
    return mode


def build_checks(sections: dict[str, dict[str, Any]], mode: Mode) -> list[dict[str, str]]:
    """The summary against the expected values: PASS, FAIL, INFO, or SKIP. A
    section that failed is a FAIL by its name."""
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    for name, section in sections.items():
        if section.get("status") == "error":
            add(name, "FAIL", f"section failed: {section.get('error', '')}")
        elif section.get("status") == "skipped":
            add(name, "SKIP", section.get("note", ""))

    model_time = sections.get("model_time", {})
    if "ratios" in model_time:
        for row in model_time["rows"]:
            if "error" in row:
                add(f"model_time {row['case']} batch {row['batch']}", "FAIL", _last_line(row["error"]))
        for size, ratios in model_time["ratios"].items():
            v3_ms = next((r["gpu_ms"] for r in model_time["rows"] if r["case"] == "v3" and str(r["batch"]) == size and "gpu_ms" in r), None)
            if mode.graphs and v3_ms is not None and int(size) in V3_REFERENCE_MS:
                add(f"v3 model time, batch {size}", "INFO", f"{v3_ms:.2f} ms; v3-cloud-profile.json had {V3_REFERENCE_MS[int(size)]:.2f}")
            if "v0_over_v3" in ratios:
                value = ratios["v0_over_v3"]
                low, high = RATIO_EXPECTED
                add(
                    f"v0/v3 model time, batch {size}",
                    verdict(low <= value <= high) if mode.graphs else "INFO",
                    f"{value:.2f}, expected {low}-{high}" + ("" if mode.graphs else " on the 5090; uncompiled here"),
                )
            if "padded_three_tensors_over_v0" in ratios:
                value = ratios["padded_three_tensors_over_v0"]
                add(
                    f"v0's bias path against padded, batch {size}",
                    verdict(value > 1.0) if mode.graphs else "INFO",
                    f"padded takes {value:.2f} times v0, expected above 1",
                )
            if "8heads_over_v0" in ratios:
                add(
                    f"8 heads against 4, batch {size}", "INFO",
                    f"{ratios['8heads_over_v0']:.2f} times on v0's path; about 1.10 on the CPU",
                )
        for name in ("padded_matches_v0", "three_tensors_match_v0"):
            if name in model_time:
                add(f"diagnostic {name}", verdict(differences_within_limits(model_time[name])),
                    f"logit max {model_time[name]['logit_max_abs']:.2e}")

    compile_section = sections.get("compile", {})
    if "graph_breaks" in compile_section:
        breaks = compile_section["graph_breaks"]
        add("graph breaks in warmup", verdict(not breaks), json.dumps(breaks) if breaks else "none")
        add(
            "compiled graphs after warmup", verdict(compile_section["unique_graphs"] <= 2),
            f"{compile_section['unique_graphs']} unique, expected 2 (batch 1 and the rest)",
        )

    launches = sections.get("launches", {})
    if "pairs" in launches:
        if "compiles_after_warmup" in launches:
            add("recompiles across launches", verdict(launches["compiles_after_warmup"] == 0),
                f"{launches['compiles_after_warmup']}, expected 0 as N and B vary")
        unsettled = [key for key, pair in launches["pairs"].items() if not pair["settled"]]
        add(
            "latency settles at every (B, K)", verdict(not unsettled) if mode.graphs else "INFO",
            f"{len(launches['pairs']) - len(unsettled)} of {len(launches['pairs'])}"
            + (f"; not: {', '.join(unsettled[:6])}" if unsettled else ""),
        )
        if "recorded_once_each" in launches:
            add(
                "each (B, K) records one graph", verdict(launches["recorded_once_each"]),
                f"record launch numbers: {json.dumps(launches['record_launch_histogram'])}; "
                "2.9.1's source says 1 at K = 256, whose warmup ran, and 2 elsewhere",
            )
        trees = launches.get("graph_trees", {})
        if "recorded_nodes" in trees:
            add(
                "recorded graphs against pairs",
                verdict(trees["recorded_nodes"] == len(launches["pairs"])),
                f"{trees['recorded_nodes']} recorded, {len(launches['pairs'])} pairs",
            )
        elif mode.graphs:
            add("recorded graphs against pairs", "INFO", json.dumps(trees))
        else:
            add("CUDA graph record and replay", "SKIP", "CUDA only")
        if mode.compiled:
            difference = launches["against_eager"]
            add(
                "compiled against eager, launches 1-3", verdict(differences_within_limits(difference)),
                _difference_line(difference),
            )

    replay = sections.get("replay", {})
    if "against_eager" in replay and mode.compiled:
        add("compiled against eager, replay", verdict(differences_within_limits(replay["against_eager"])),
            _difference_line(replay["against_eager"]))
        add("filler changes nothing, across K", verdict(differences_within_limits(replay["filler_invariance"])),
            _difference_line(replay["filler_invariance"]))
        sync = replay["weight_sync"]
        moved = sync["replay_against_before"]["logit_max_abs"]
        add(
            "replay follows a weight sync",
            verdict(differences_within_limits(sync["replay_against_eager"]) and moved > LOGIT_MAX_ABS),
            f"{_difference_line(sync['replay_against_eager'])}; moved {moved:.3f} from before",
        )
    elif "against_eager" in replay:
        add("compiled against eager", "SKIP", "uncompiled mode")

    memory = sections.get("memory", {})
    if "samples" in memory:
        growth = memory["growth_mb"]
        if mode.graphs:
            stable = (
                growth["reserved_mb"] <= MEMORY_GROWTH_MB
                and -growth["device_free_mb"] <= MEMORY_GROWTH_MB
                and memory["new_captures"] == 0
            )
            add(
                f"memory over {memory['launches']:,} launches", verdict(stable),
                f"reserved {growth['reserved_mb']:+.0f} MB, device free {growth['device_free_mb']:+.0f} MB, "
                f"{memory['new_captures']} new graphs, {memory['new_compiles']} new compiles",
            )
        else:
            add("memory growth", "SKIP", "needs CUDA graphs; ran as a harness check")

    train = sections.get("train_steps", {})
    if "shapes" in train:
        for name, shape in train["shapes"].items():
            rate = shape["steps_per_second"]
            add(
                f"probe training, {name}",
                verdict(rate >= MIN_PROBE_STEPS_PER_SECOND) if mode.graphs else "INFO",
                f"{rate:.1f} steps/s at batch {train['batch']}, {shape['hours_for_40k_steps']:.1f} h for 40K",
            )
    return checks


def _difference_line(difference: dict[str, float]) -> str:
    return (
        f"logits max {difference['logit_max_abs']:.2e}, values max {difference['value_max_abs']:.2e}, "
        f"policy TV {difference['policy_max_tv']:.2e}, "
        f"{difference['logit_outside_tolerance']:.0f} of {difference['logit_count']:.0f} logits past 0.003 + 3%"
    )


def print_summary(checks: list[dict[str, str]], output: Path) -> None:
    print("\nSummary")
    for check in checks:
        print(f"  {check['status']:<4}  {check['name']}: {check['detail']}")
    failed = sum(check["status"] == "FAIL" for check in checks)
    print(f"\n{failed} failed. Send {output} back." if failed else f"\nNo failures. Send {output} back.")


def _source_hashes() -> dict[str, str]:
    paths = ("nn/transformer-1889.py", "nn/transformer-v3.py", "train/eval_server.py", "train/gpu/nvidia.py", "cloud/profile_1889.py")
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="CPU, uncompiled unless --compile; CUDA parts skipped")
    parser.add_argument("--rocm", action="store_true", help="ROCm, uncompiled, fp32, small batches; CUDA parts skipped")
    parser.add_argument("--compile", action="store_true", help="With --cpu: the compile path, without graphs")
    parser.add_argument("--smoke", action="store_true", help="A tiny model and few launches, to check the script")
    parser.add_argument("--sections", default=",".join(ALL_SECTIONS), help="Comma-separated sections to run")
    parser.add_argument("--output", help="The JSON to write (default: 1889-cloud-profile.json on the 5090)")
    args = parser.parse_args(argv)
    sections = args.sections.split(",")
    unknown = set(sections) - set(ALL_SECTIONS)
    if unknown:
        _fail(f"unknown --sections {sorted(unknown)}; they are {', '.join(ALL_SECTIONS)}")

    mode = _mode_from_args(args)
    output = Path(args.output or mode.output)
    try:
        config = _model_config(TrainingConfig.from_json(Path(V0_CONFIG).read_text()), mode)
        v3_config = TrainingConfig.from_json(Path(V3_CONFIG).read_text())
        game, v3_game = get_game(config), get_game(v3_config)
    except Exception as exc:
        _fail(
            f"cannot build the games ({type(exc).__name__}: {exc}). Build the extensions: "
            "python setup.py build_ext --inplace (after a pull: python setup.py clean_build)"
        )

    report: dict[str, Any] = {
        "revision": REVISION,
        "mode": mode.name,
        "mode_settings": {k: v for k, v in vars(mode).items()},
        "smoke": args.smoke,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,  # type: ignore[attr-defined]
        "hip": getattr(torch.version, "hip", None),
        "gpu": torch.cuda.get_device_name() if mode.device == "cuda" else None,
        "git_commit": _git_commit(),
        "source_sha256": _source_hashes(),
        "config": json.loads(config.to_json()),
        "input_note": "reachable random-play states; not measured MCTS traffic",
        "weights": "random initialized",
        "sections": {},
    }

    def save() -> None:
        output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    save()
    if mode.device == "cuda":
        gpu = detect_gpu("cuda")
        report["gpu_optimizations"] = gpu.apply_optimizations()
        report["memory_at_start"] = memory_stats(mode)
        if mode.graphs and report["memory_at_start"]["device_free_mb"] < 16_000:
            print(
                f"Warning: only {report['memory_at_start']['device_free_mb']:,.0f} MB free on the GPU; "
                "is a run still on it?", flush=True,
            )
    print(f"Mode {mode.name}, torch {torch.__version__}; writing {output}", flush=True)
    print(f"Generating real states from random playouts ({mode.pool_states})...", flush=True)
    try:
        pool = make_pool(game, config, mode.pool_states)
    except Exception as exc:
        report["fatal"] = traceback.format_exc()
        save()
        _fail(f"random playouts failed ({type(exc).__name__}: {exc}); see {output}")
    report["pool"] = {
        "states": len(pool),
        "players": {str(n): sum(e.players == n for e in pool) for n in config.iter_player_counts()},
        "phases": {
            name: sum(e.phase == i for e in pool) for i, name in enumerate(game.policy.phase_names)
        },
        "records": {
            stat: float(fn([e.record_count for e in pool]))
            for stat, fn in (("min", np.min), ("mean", np.mean), ("median", np.median), ("max", np.max))
        },
    }
    by_k = {k: [e for e in pool if e.record_count <= k] for k in mode.ks}
    empty = [k for k, states in by_k.items() if not states]
    if empty:
        _fail(f"no playout state fits K = {empty}")
    ctx = Context(mode, config, v3_config, game, v3_game, pool, by_k, save)

    runners: dict[str, Callable[[Context], dict[str, Any]]] = {
        "model_time": section_model_time,
        "compile": section_compile,
        "launches": section_launches,
        "replay": section_replay,
        "memory": section_memory,
        "train_steps": section_train_steps,
    }
    for name in sections:
        print(f"{name}...", flush=True)
        begin = time.perf_counter()
        section: dict[str, Any]
        try:
            if name in ("launches", "replay", "memory") and mode.compiled and ctx.serving is None:
                raise RuntimeError("the compile section did not produce a model")
            section = runners[name](ctx)
        except Exception as exc:
            section = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            print(f"  failed: {section['error']}", flush=True)
        section["seconds"] = time.perf_counter() - begin
        report["sections"][name] = section
        save()
    checks = build_checks(report["sections"], mode)
    report["checks"] = checks
    save()
    print_summary(checks, output)
    return 1 if any(check["status"] == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
