"""The capacity probe (1889/model-design.md §5.7): training only, on a saved buffer.

Trains each candidate size from scratch on a saved 1889 replay buffer with the
real ``Trainer``, under one schedule, and scores it on held-out rows by value
MSE and policy KL (cross-entropy minus target entropy), overall, by player
count, and for TRACK, TILE, STOCK, and BUY_TRAIN. It prints a table and writes
``probe.json``, with the verdict by the section's criterion.

**The held-out rows** are the newest 2 epochs' games. The buffer's epoch record
gives the first game id of the checkpoint's second-newest epoch, and each row
its game id, so the cut is that game's first row. A buffer saved without the
record (1889-v0's) holds out the newest ``2 × games_per_epoch`` games instead:
its ring index gives write order, a row's state shows whether it is its game's
first decision (``Game.game_start_rows``), and every game goes in whole, so an
epoch is its games' rows and the cut falls between games.
TensorBoard's ``self_play_aggregate/total_examples`` for the checkpoint's last
two epochs must give the same row count; the probe reports whether it does. They
differ by a partial epoch at the buffer's tail when a graceful shutdown
stopped mid-epoch: those rows hold no TensorBoard count, and join the held-out
newest games. The trainer's buffer holds only the older rows, so sampling,
priorities included, cannot reach a held-out row.

**Outputs** go under ``--out``: for each model, ``<role>/checkpoint_epoch_NNNN.pt``
in the run's checkpoint format, at the run's epoch and step and with the run's
config at the model's size, for ``python -m train --resume`` to warm-start from,
and ``<role>/result.json``. A model with a result is skipped; an interrupted one
continues from its ``<role>/progress.pt``. Rerun the same command to continue.

Run from the repo root: ``python -m train.capacity_probe --checkpoint
checkpoints/1889-v0/checkpoint_epoch_0050.pt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from games import get_game
from nn import create_model
from train.checkpoint import load_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.gpu import detect_gpu
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer

# The criterion's constants (§5.7).
GROW_MARGIN_GAPS = 2.0
GROW_MARGIN_SHARE = 0.03
SHRINK_EPOCH = 150

MEASURES = ("value_mse", "policy_kl")
PHASE_GROUPS = {"TRACK": "track", "TILE": "tile", "STOCK": "stock", "BUY_TRAIN": "train"}
SAMPLES_TAG = "self_play_aggregate/total_examples"
EXPECTED_GAME = "1889"


class ProbeError(Exception):
    """Something the user must fix; main prints it without a traceback."""


@dataclass(frozen=True)
class ProbeModel:
    role: str  # base_seed0, base_seed1, small, large, heads8
    d_model: int
    num_layers: int
    num_heads: int
    seed: int

    @property
    def shape(self) -> str:
        return f"{self.d_model}x{self.num_layers}, {self.num_heads} heads"


MODELS = (
    ProbeModel("base_seed0", 256, 15, 4, 0),
    ProbeModel("base_seed1", 256, 15, 4, 1),
    ProbeModel("small", 192, 12, 4, 0),
    ProbeModel("large", 384, 18, 6, 0),
    ProbeModel("heads8", 256, 15, 8, 0),
)
# --smoke's: the script's check, at sizes for a shared GPU. It says nothing of capacity.
SMOKE_MODELS = (
    ProbeModel("base_seed0", 32, 2, 4, 0),
    ProbeModel("base_seed1", 32, 2, 4, 1),
    ProbeModel("small", 24, 1, 4, 0),
    ProbeModel("large", 48, 3, 6, 0),
    ProbeModel("heads8", 32, 2, 8, 0),
)


# ---------------------------------------------------------------------------
# The held-out split
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoldoutSplit:
    train_rows: int  # the oldest rows
    holdout_rows: int  # the newest
    games: int
    tensorboard_rows: int | None  # the same epochs' row count, if logged
    agrees_with_tensorboard: bool | None


def tensorboard_rows(
    examples: list[tuple[int, float]], last_epoch: int, holdout_epochs: int,
) -> int | None:
    """The logged rows of the ``holdout_epochs`` epochs through ``last_epoch``
    (1-indexed), or None unless all of them are logged."""
    by_epoch = dict(examples)
    wanted = range(last_epoch - holdout_epochs + 1, last_epoch + 1)
    if not all(epoch in by_epoch for epoch in wanted):
        return None
    return round(sum(by_epoch[epoch] for epoch in wanted))


def choose_split(
    game_starts: np.ndarray,
    total_rows: int,
    *,
    games_per_epoch: int,
    holdout_epochs: int,
    logged_rows: int | None,
) -> HoldoutSplit:
    """Cut between games, holding out the newest ``holdout_epochs`` epochs' games."""
    games = games_per_epoch * holdout_epochs
    if len(game_starts) < games:
        raise ProbeError(
            f"the buffer holds {len(game_starts)} games' starts, but holding out {holdout_epochs} "
            f"epochs takes {games} games; lower --holdout-epochs"
        )
    first = int(game_starts[-games])
    holdout = total_rows - first
    return HoldoutSplit(
        train_rows=first,
        holdout_rows=holdout,
        games=games,
        tensorboard_rows=logged_rows,
        agrees_with_tensorboard=None if logged_rows is None else logged_rows == holdout,
    )


def record_split(
    game_ids: np.ndarray,
    epochs: list[tuple[int, int]],
    *,
    last_epoch: int,
    holdout_epochs: int,
    logged_rows: int | None,
) -> HoldoutSplit | None:
    """The cut from the buffer's epoch record: the first game of the epoch
    ``holdout_epochs`` back from ``last_epoch`` (1-indexed, the checkpoint's).

    ``game_ids`` are the rows', oldest first, and ``epochs`` the record's
    ``(epoch, first game id)``, epochs 0-indexed. None when the record lacks
    that epoch, as for a buffer saved without it.
    """
    first_epoch = last_epoch - holdout_epochs
    first_game = next((start for epoch, start in reversed(epochs) if epoch == first_epoch), None)
    if first_game is None:
        return None
    first = int(np.count_nonzero(game_ids < first_game))
    holdout = len(game_ids) - first
    return HoldoutSplit(
        train_rows=first,
        holdout_rows=holdout,
        games=len(np.unique(game_ids[first:])),
        tensorboard_rows=logged_rows,
        agrees_with_tensorboard=None if logged_rows is None else logged_rows == holdout,
    )


# ---------------------------------------------------------------------------
# Held-out scores
# ---------------------------------------------------------------------------

def group_rows(
    player_counts: np.ndarray, phase_ids: np.ndarray, phase_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Each reporting group's rows, as boolean masks."""
    groups = {"all": np.ones(len(player_counts), dtype=bool)}
    for n in sorted(set(player_counts.tolist())):
        groups[f"{n}p"] = player_counts == n
    for label, name in PHASE_GROUPS.items():
        groups[label] = phase_ids == phase_names.index(name)
    return groups


def summarize_scores(
    cross_entropy: np.ndarray,
    target_entropy: np.ndarray,
    value_sqerr: np.ndarray,
    player_counts: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Value MSE over real player slots and mean policy KL, for each group.

    A value's denominator is its group's real players, as ``train_step``'s; rows
    count uniformly, since a held-out row was never prioritized.
    """
    result: dict[str, dict[str, float]] = {}
    for name, rows in groups.items():
        count = int(rows.sum())
        if count == 0:
            continue
        result[name] = {
            "rows": count,
            "value_mse": float(value_sqerr[rows].sum() / player_counts[rows].astype(np.float64).sum()),
            "policy_kl": float((cross_entropy[rows].astype(np.float64) - target_entropy[rows]).mean()),
        }
    return result


def evaluate_holdout(
    trainer: Trainer, rows: dict[str, np.ndarray], batch_size: int, phase_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Score every held-out row by the trainer's input path and loss functions."""
    was_training = trainer.model.training
    trainer.model.eval()
    total = len(rows["states"])
    scores: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, total, batch_size):
        stop = start + batch_size
        scores.append(trainer.score_rows(
            rows["states"][start:stop], rows["legal_masks"][start:stop],
            rows["policy_targets"][start:stop], rows["value_targets"][start:stop],
            rows["player_counts"][start:stop],
        ))
    trainer.model.train(was_training)
    cross_entropy, target_entropy, value_sqerr = (np.concatenate(part) for part in zip(*scores))
    groups = group_rows(rows["player_counts"], rows["phase_ids"], phase_names)
    return summarize_scores(cross_entropy, target_entropy, value_sqerr, rows["player_counts"], groups)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def _compare(candidate: float, seed_a: float, seed_b: float) -> dict[str, Any]:
    """A measure of a candidate against 256 × 15: the two seeds' mean and gap.

    Lower is better. ``beats`` is the grow margin: better by more than twice the
    gap and by at least 3%. ``within_gap`` is no worse than the mean by more than
    the gap; a smaller net that wins by more still shows v0 is larger than needed.
    """
    baseline = (seed_a + seed_b) / 2
    gap = abs(seed_a - seed_b)
    margin = baseline - candidate
    return {
        "candidate": candidate,
        "baseline": baseline,
        "gap": gap,
        "margin": margin,
        "margin_share": margin / baseline if baseline > 0 else None,
        "beats": margin > GROW_MARGIN_GAPS * gap and margin >= GROW_MARGIN_SHARE * baseline,
        "within_gap": -margin <= gap,
    }


def decide_verdict(results: dict[str, dict[str, float]], epoch: int) -> dict[str, Any]:
    """§5.7's criterion on each model's overall measures: grow, shrink, 8 heads,
    or keep, the first that holds. ``epoch`` is the run's, 1-indexed.

    ``results`` maps a role to its ``value_mse`` and ``policy_kl``. A missing
    model leaves the verdict ``incomplete``.
    """
    roles = ("base_seed0", "base_seed1", "small", "large", "heads8")
    missing = [role for role in roles if role not in results]
    if missing:
        return {"verdict": "incomplete", "missing": missing, "reason": f"no results for {', '.join(missing)}"}
    comparisons = {
        role: {
            measure: _compare(
                results[role][measure], results["base_seed0"][measure], results["base_seed1"][measure],
            )
            for measure in MEASURES
        }
        for role in ("small", "large", "heads8")
    }
    grow = all(c["beats"] for c in comparisons["large"].values())
    near = all(c["within_gap"] for c in comparisons["small"].values())
    shrink = near and epoch >= SHRINK_EPOCH
    heads8 = all(c["beats"] for c in comparisons["heads8"].values())
    if grow:
        verdict = "grow"
        reason = "384 x 18 beats 256 x 15 on both measures by more than twice the seed gap and at least 3%"
    elif shrink:
        verdict = "shrink"
        reason = f"at epoch {epoch}, 192 x 12 is within the seed gap of 256 x 15 on both measures"
    elif heads8:
        verdict = "8 heads"
        reason = "8 heads beat 4 on both measures by more than twice the seed gap and at least 3%"
    else:
        verdict = "keep"
        reason = "no candidate meets its margin"
    notes = []
    if near and not shrink:
        notes.append(f"192 x 12 is within the seed gap, which counts only at epoch {SHRINK_EPOCH}")
    if heads8 and verdict != "8 heads":
        notes.append("8 heads also beat 4 by the margin")
    if shrink and verdict != "shrink":
        notes.append("192 x 12 is also within the seed gap")
    return {
        "verdict": verdict,
        "reason": reason,
        "notes": notes,
        "conditions": {"grow": grow, "shrink": shrink, "8 heads": heads8},
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------

def probe_config(
    run_config: TrainingConfig, model: ProbeModel, *, steps: int, batch_size: int, warmup_steps: int,
) -> TrainingConfig:
    """The trainer's config: the run's optimizer and rates at this model's size,
    with its one LR schedule spanning the probe's steps."""
    return dataclasses.replace(
        sized_config(run_config, model),
        batch_size=batch_size,
        min_buffer_size=max(run_config.min_buffer_size, batch_size),
        training_steps_per_epoch=steps,
        lr_decay_end_epoch=1,
        warmup_epochs=warmup_steps / steps,
        seed=model.seed,
    )


def sized_config(run_config: TrainingConfig, model: ProbeModel) -> TrainingConfig:
    """The run's config at this model's size: what its weights resume under."""
    return dataclasses.replace(
        run_config, d_model=model.d_model, num_layers=model.num_layers, num_heads=model.num_heads,
    )


def _save_atomic(save: Any, path: Path) -> None:
    """Write through a temporary name, so an interrupted save leaves the last file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    save(temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _save_atomic(lambda p: p.write_text(json.dumps(value, indent=2) + "\n"), path)


def _warm_start_trainer_state(
    model: torch.nn.Module, config: TrainingConfig, device: torch.device, run_step: int,
) -> dict[str, object]:
    """A fresh optimizer's state at the run's position on its own LR schedule,
    so ``--resume`` carries on at the run's rate rather than the probe's."""
    trainer = Trainer(model, config, device)
    trainer.scheduler.last_epoch = run_step
    if trainer._aux_scheduler is not None:
        trainer._aux_scheduler.last_epoch = run_step
    trainer._global_step = run_step
    return trainer.state_dict()


def train_model(
    model_spec: ProbeModel,
    run_config: TrainingConfig,
    train_buffer: ReplayBuffer,
    holdout: dict[str, np.ndarray],
    *,
    out_dir: Path,
    steps: int,
    batch_size: int,
    warmup_steps: int,
    eval_batch_size: int,
    save_every: int,
    eval_every: int,
    log_every: int,
    device: torch.device,
    run_epoch: int,
    run_step: int,
    run_rng_state: dict[str, object] | None,
    signature: dict[str, Any],
) -> dict[str, Any]:
    """Train and score one model, or return its finished result."""
    model_dir = out_dir / model_spec.role
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / "result.json"
    progress_path = model_dir / "progress.pt"
    model_signature = {**signature, "model": dataclasses.asdict(model_spec)}
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["signature"] != model_signature:
            raise ProbeError(
                f"{result_path} came from other settings; pass another --out, or delete it to retrain"
            )
        print(f"{model_spec.role} ({model_spec.shape}): already done, skipping")
        return result

    config = probe_config(run_config, model_spec, steps=steps, batch_size=batch_size, warmup_steps=warmup_steps)
    torch.manual_seed(model_spec.seed)
    model = create_model(config).to(device)
    trainer = Trainer(model, config, device)
    rng = np.random.default_rng(model_spec.seed)
    train_seconds = 0.0
    history: list[dict[str, Any]] = []
    if progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress["signature"] != model_signature:
            raise ProbeError(
                f"{progress_path} came from other settings; pass another --out, or delete it to restart"
            )
        model.load_state_dict(progress["model_state_dict"])
        trainer.load_state_dict(progress["trainer_state"])
        rng.bit_generator.state = progress["rng_state"]
        train_seconds = progress["train_seconds"]
        history = progress["history"]
        print(f"{model_spec.role} ({model_spec.shape}): continuing at step {trainer.global_step:,}")
    else:
        print(f"{model_spec.role} ({model_spec.shape}): training {steps:,} steps")

    def save_progress(path: Path) -> None:
        torch.save({
            "signature": model_signature,
            "model_state_dict": model.state_dict(),
            "trainer_state": trainer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "train_seconds": train_seconds,
            "history": history,
        }, path)

    phase_names = tuple(get_game(config).policy.phase_names)
    recent: deque[dict[str, float]] = deque(maxlen=100)
    model.train()
    while trainer.global_step < steps:
        begin = time.perf_counter()
        losses = trainer.train_step(train_buffer, batch_size, rng)
        train_seconds += time.perf_counter() - begin
        recent.append({"policy_kl": losses["policy_kl"], "value_loss": losses["value_loss"]})
        step = trainer.global_step
        if step % log_every == 0 or step == steps:
            print(
                f"  step {step:,}/{steps:,}: policy KL {losses['policy_kl']:.4f}, "
                f"value MSE {losses['value_loss']:.4f}, lr {trainer.lr:.2e}, "
                f"{step_rate(step, train_seconds):.1f} steps/s",
                flush=True,
            )
        if eval_every and step % eval_every == 0 and step != steps:
            scores = evaluate_holdout(trainer, holdout, eval_batch_size, phase_names)
            history.append({"step": step, **{m: scores["all"][m] for m in MEASURES}})
            print(f"    held out at {step:,}: {_overall_line(scores)}", flush=True)
        if save_every and step % save_every == 0 and step != steps:
            _save_atomic(save_progress, progress_path)

    scores = evaluate_holdout(trainer, holdout, eval_batch_size, phase_names)
    print(f"  held out: {_overall_line(scores)}", flush=True)
    result: dict[str, Any] = {
        "signature": model_signature,
        "role": model_spec.role,
        "shape": model_spec.shape,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": trainer.global_step,
        "steps_per_second": step_rate(trainer.global_step, train_seconds),
        "train_seconds": train_seconds,
        # The last steps' training measures, against the held-out ones: their
        # distance is the fit to the training rows.
        "train_recent": {
            name: float(np.mean([row[name] for row in recent])) for name in ("policy_kl", "value_loss")
        } if recent else None,
        "holdout": scores,
        "history": history,
    }

    # The weights, for --resume: the run's config at this size, at its epoch and step.
    resume_config = sized_config(run_config, model_spec)
    checkpoint_path = model_dir / f"checkpoint_epoch_{run_epoch:04d}.pt"
    trainer_state = _warm_start_trainer_state(model, resume_config, device, run_step)
    _save_atomic(lambda path: save_checkpoint(
        path=path,
        epoch=run_epoch - 1,
        model=model,
        trainer_state=trainer_state,
        config=resume_config,
        metrics={m: scores["all"][m] for m in MEASURES},
        buffer_stats={"size": len(train_buffer), "capacity": run_config.buffer_capacity},
        rng_state=run_rng_state,
    ), checkpoint_path)
    result["checkpoint"] = str(checkpoint_path)
    _write_json(result_path, result)
    progress_path.unlink(missing_ok=True)
    return result


def step_rate(steps: int, seconds: float) -> float:
    return steps / seconds if seconds > 0 else 0.0


def _overall_line(scores: dict[str, dict[str, float]]) -> str:
    return f"value MSE {scores['all']['value_mse']:.4f}, policy KL {scores['all']['policy_kl']:.4f}"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def format_tables(results: dict[str, dict[str, Any]]) -> str:
    """One table a measure: models by groups."""
    groups = ["all", "3p", "4p", "5p", *PHASE_GROUPS]
    lines = []
    for measure in MEASURES:
        lines.append(f"\n{measure}")
        lines.append(f"  {'model':<26}" + "".join(f"{group:>10}" for group in groups))
        for role, result in results.items():
            cells = (
                f"{result['holdout'][group][measure]:>10.4f}" if group in result["holdout"] else f"{'-':>10}"
                for group in groups
            )
            lines.append(f"  {role + ' (' + result['shape'].replace(' heads', 'h') + ')':<26}" + "".join(cells))
    lines.append(
        "\n  " + "; ".join(f"{role}: {r['steps_per_second']:.1f} steps/s" for role, r in results.items())
    )
    return "\n".join(lines)


def format_verdict(verdict: dict[str, Any]) -> str:
    lines = [f"\nVerdict: {verdict['verdict'].upper()} ({verdict['reason']})"]
    for note in verdict.get("notes", ()):
        lines.append(f"  Note: {note}")
    for role, by_measure in verdict.get("comparisons", {}).items():
        for measure, c in by_measure.items():
            share = "" if c["margin_share"] is None else f" ({100 * c['margin_share']:+.2f}%)"
            lines.append(
                f"  {role} {measure}: {c['candidate']:.4f} against {c['baseline']:.4f}, "
                f"margin {c['margin']:+.4f}{share}, seed gap {c['gap']:.4f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="The run's checkpoint; gives its config, epoch, and step")
    parser.add_argument("--buffer", help="Replay buffer directory (default: replay_buffer beside the checkpoint)")
    parser.add_argument("--logdir", help="TensorBoard directory (default: the run config's tensorboard_dir)")
    parser.add_argument("--out", help="Output directory (default: probe_eNNNN beside the checkpoint)")
    parser.add_argument("--steps", type=int, default=40_000, help="Training steps per model (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch (default: %(default)s)")
    parser.add_argument("--warmup-steps", type=int, help="LR warmup (default: 1000, or a tenth of --steps if less)")
    parser.add_argument("--eval-batch-size", type=int, help="Held-out batch (default: --batch-size)")
    parser.add_argument("--holdout-epochs", type=int, default=2, help="Newest epochs held out (default: %(default)s)")
    parser.add_argument("--save-every", type=int, default=2_000, help="Steps between progress saves (default: %(default)s)")
    parser.add_argument("--eval-every", type=int, default=10_000, help="Steps between held-out scores for the curve, 0 for none (default: %(default)s)")
    parser.add_argument("--log-every", type=int, default=1_000, help="Steps between progress lines (default: %(default)s)")
    parser.add_argument("--device", help="Device (default: cuda if available)")
    parser.add_argument("--models", help="Comma-separated roles to run (default: all; the verdict needs all)")
    parser.add_argument("--smoke", action="store_true", help="Small models, to check the script; says nothing of capacity")
    return parser


def _load_rows(
    run_config: TrainingConfig, buffer_dir: Path, *, capacity: int,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    """The saved buffer's rows, oldest first, and its epoch record."""
    game = get_game(run_config)
    max_players = run_config.effective_max_players
    saved = ReplayBuffer(
        game, capacity, game.state_size(max_players), max_players,
        min_players=run_config.effective_min_players, max_players=max_players,
    )
    if saved.load(buffer_dir) == 0:
        raise ProbeError(f"no replay buffer loaded from {buffer_dir}; see the warning above, or pass --buffer")
    return saved.rows_oldest_first(), saved.epochs


def _train_buffer(run_config: TrainingConfig, rows: dict[str, np.ndarray], count: int) -> ReplayBuffer:
    """A buffer of the oldest ``count`` rows only, with their priorities and
    games, under the run's prioritized replay. It samples all of them."""
    game = get_game(run_config)
    max_players = run_config.effective_max_players
    buffer = ReplayBuffer(
        game, count, game.state_size(max_players), max_players,
        min_players=run_config.effective_min_players, max_players=max_players,
        priority_fraction=run_config.replay_priority_fraction,
        priority_exponent=run_config.replay_priority_exponent,
    )
    buffer.add_stacked(
        rows["states"][:count], rows["phase_ids"][:count], rows["legal_masks"][:count],
        rows["policy_targets"][:count], rows["value_targets"][:count],
        player_counts=rows["player_counts"][:count], priorities=rows["priorities"][:count],
        game_ids=rows["game_ids"][:count], flags=rows["flags"][:count], net_worths=rows["net_worths"][:count],
    )
    return buffer


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ProbeError(f"no checkpoint {checkpoint_path}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cp = load_checkpoint(checkpoint_path, torch.device("cpu"))
    run_config = TrainingConfig.from_json(cp["config_json"])
    if run_config.game != EXPECTED_GAME:
        raise ProbeError(f"the probe is 1889's, and {checkpoint_path} plays {run_config.game!r}")
    run_epoch = cp["epoch"] + 1
    step = cp["trainer_state"]["global_step"]
    assert isinstance(step, int)
    buffer_dir = Path(args.buffer) if args.buffer else checkpoint_path.parent / "replay_buffer"
    out_dir = Path(args.out) if args.out else checkpoint_path.parent / f"probe_e{run_epoch:04d}"
    logdir = args.logdir or run_config.tensorboard_dir
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else min(1_000, args.steps // 10)
    eval_batch_size = args.eval_batch_size or args.batch_size

    models = SMOKE_MODELS if args.smoke else MODELS
    if args.models:
        wanted = args.models.split(",")
        unknown = set(wanted) - {m.role for m in models}
        if unknown:
            raise ProbeError(f"unknown --models {sorted(unknown)}; roles are {[m.role for m in models]}")
        models = tuple(m for m in models if m.role in wanted)

    metadata_path = buffer_dir / "metadata.json"
    if not metadata_path.exists():
        raise ProbeError(f"no replay buffer at {buffer_dir}; pass --buffer")
    print(f"Loading {buffer_dir}...", flush=True)
    capacity = json.loads(metadata_path.read_text())["capacity"]
    rows, epochs = _load_rows(run_config, buffer_dir, capacity=capacity)
    total = len(rows["states"])

    try:
        from train.tb_reader import read_tb_scalars
        examples = read_tb_scalars(logdir).get(SAMPLES_TAG, [])
    except ImportError:
        examples = []
    logged_rows = tensorboard_rows(examples, run_epoch, args.holdout_epochs)
    split = record_split(
        rows["game_ids"], epochs,
        last_epoch=run_epoch, holdout_epochs=args.holdout_epochs, logged_rows=logged_rows,
    )
    if split is None:
        # No epoch record: game boundaries from the rows' states.
        find_game_starts = get_game(run_config).game_start_rows
        assert find_game_starts is not None
        split = choose_split(
            find_game_starts(rows["states"], run_config.effective_max_players), total,
            games_per_epoch=run_config.games_per_epoch,
            holdout_epochs=args.holdout_epochs,
            logged_rows=logged_rows,
        )
    if split.train_rows < args.batch_size:
        raise ProbeError(
            f"{split.train_rows} rows are left to train on, under a batch of {args.batch_size}"
        )
    holdout = {name: array[split.train_rows:].copy() for name, array in rows.items()}
    train_buffer = _train_buffer(run_config, rows, split.train_rows)
    del rows

    passes = args.steps * args.batch_size / split.train_rows
    epochs_of_training_rows = split.train_rows * args.holdout_epochs / split.holdout_rows
    print(
        f"Run epoch {run_epoch}: {total:,} rows; held out the newest {split.holdout_rows:,} "
        f"({split.games} games); training on {split.train_rows:,}, about "
        f"{epochs_of_training_rows:.1f} epochs' rows"
    )
    print(f"  {args.steps:,} steps at batch {args.batch_size} make {passes:.1f} passes over the training rows")
    if split.agrees_with_tensorboard is None:
        print(f"  TensorBoard has no {SAMPLES_TAG} for these epochs in {logdir}: the count is the buffer's alone")
    elif split.agrees_with_tensorboard:
        print(f"  TensorBoard agrees: {split.tensorboard_rows:,} rows in these epochs")
    else:
        print(
            f"  TensorBoard counts {split.tensorboard_rows:,} rows in epochs {run_epoch - args.holdout_epochs + 1}-"
            f"{run_epoch}. A partial epoch from a graceful shutdown makes this differ: the cut still falls "
            f"between games, on the newest {split.games}. Otherwise check that this buffer is this checkpoint's."
        )

    detect_gpu(device.type).apply_optimizations()
    out_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "checkpoint_epoch": run_epoch,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "warmup_steps": warmup_steps,
        "train_rows": split.train_rows,
        "holdout_rows": split.holdout_rows,
        "learning_rate": run_config.learning_rate,
        "lr_min": run_config.lr_min,
        "optimizer": run_config.optimizer,
    }
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for model_spec in models:
        try:
            results[model_spec.role] = train_model(
                model_spec, run_config, train_buffer, holdout,
                out_dir=out_dir,
                steps=args.steps,
                batch_size=args.batch_size,
                warmup_steps=warmup_steps,
                eval_batch_size=eval_batch_size,
                save_every=args.save_every,
                eval_every=args.eval_every,
                log_every=args.log_every,
                device=device,
                run_epoch=run_epoch,
                run_step=step,
                run_rng_state=cp.get("rng_state"),
                signature=signature,
            )
        except RuntimeError as exc:  # NaN loss or out of memory: the others can still finish.
            failures[model_spec.role] = traceback.format_exc()
            print(f"{model_spec.role} failed: {exc}", flush=True)
        # A trainer's hook on its model makes a cycle: free it before the next model.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    verdict = decide_verdict(
        {role: result["holdout"]["all"] for role, result in results.items()}, run_epoch,
    )
    report = {
        "run_epoch": run_epoch,
        "smoke": args.smoke,
        "torch": torch.__version__,
        "device": str(device),
        "buffer": str(buffer_dir),
        "split": dataclasses.asdict(split),
        "passes_over_training_rows": passes,
        "settings": signature,
        "results": results,
        "failures": failures,
        "verdict": verdict,
    }
    report_path = out_dir / "probe.json"
    _write_json(report_path, report)
    print(format_tables(results))
    print(format_verdict(verdict))
    print(f"\nReport: {report_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        run_probe(args)
    except ProbeError as exc:
        sys.exit(f"capacity_probe: {exc}")
    except KeyboardInterrupt:
        sys.exit("capacity_probe: interrupted. Rerun the same command to continue from the last save.")


if __name__ == "__main__":
    main()
