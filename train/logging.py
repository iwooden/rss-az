"""Tensorboard integration and Rich live terminal UI for training."""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch.utils.tensorboard.writer import SummaryWriter

if TYPE_CHECKING:
    from train.config import TrainingConfig


def _format_duration(secs: float) -> str:
    """Format seconds as 'Xh Ym Zs', omitting zero-valued leading components."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    if secs < 1.0:
        return f"{secs:.1f}s"
    return f"{s}s"


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    """Build a text progress bar: ████████░░░░░░░░ 45.0%"""
    frac = done / max(total, 1)
    filled = int(width * frac)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"{bar} {frac * 100:5.1f}%"


class TrainingLogger:
    """Tensorboard + Rich live terminal output for training."""

    def __init__(self, tensorboard_dir: str) -> None:
        self.writer = SummaryWriter(tensorboard_dir)
        self.console = Console()
        self._live: Live | None = None
        self._phase_start: float = 0.0

        # Self-play state for panel building
        self._sp_epoch = 0
        self._sp_num_epochs = 0
        self._sp_total_games = 0
        self._sp_games_done = 0
        self._sp_total_examples = 0
        self._sp_avg_moves = 0.0
        self._sp_rank_net_worths: list[float] = []
        self._sp_rank_mins: list[float] = []
        self._sp_rank_maxs: list[float] = []
        self._sp_policy_entropy: float = 0.0
        self._sp_top1_visit_frac: float = 0.0

        # Training state for panel building
        self._tr_epoch = 0
        self._tr_num_epochs = 0
        self._tr_total_steps = 0
        self._tr_step = 0
        self._tr_losses: dict[str, float] = {}
        self._tr_lr = 0.0

    # --- Live display: Self-Play ---

    def begin_self_play(
        self, epoch: int, num_epochs: int, total_games: int
    ) -> None:
        self._sp_epoch = epoch
        self._sp_num_epochs = num_epochs
        self._sp_total_games = total_games
        self._sp_games_done = 0
        self._sp_total_examples = 0
        self._sp_avg_moves = 0.0
        self._sp_rank_net_worths = []
        self._sp_rank_mins = []
        self._sp_rank_maxs = []
        self._sp_policy_entropy = 0.0
        self._sp_top1_visit_frac = 0.0
        self._phase_start = time.perf_counter()

        self._live = Live(
            self._build_self_play_panel(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def update_self_play(
        self,
        games_done: int,
        total_examples: int,
        avg_moves: float,
        rank_net_worths: list[float] | None = None,
        rank_mins: list[float] | None = None,
        rank_maxs: list[float] | None = None,
        policy_entropy: float | None = None,
        top1_visit_frac: float | None = None,
    ) -> None:
        self._sp_games_done = games_done
        self._sp_total_examples = total_examples
        self._sp_avg_moves = avg_moves
        if rank_net_worths is not None:
            self._sp_rank_net_worths = rank_net_worths
        if rank_mins is not None:
            self._sp_rank_mins = rank_mins
        if rank_maxs is not None:
            self._sp_rank_maxs = rank_maxs
        if policy_entropy is not None:
            self._sp_policy_entropy = policy_entropy
        if top1_visit_frac is not None:
            self._sp_top1_visit_frac = top1_visit_frac
        if self._live is not None:
            self._live.update(self._build_self_play_panel())

    def end_self_play(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _build_self_play_panel(self) -> Panel:
        elapsed = time.perf_counter() - self._phase_start
        bar = _progress_bar(self._sp_games_done, self._sp_total_games)
        rate = self._sp_games_done / max(elapsed, 0.01) * 60
        lines = Text()
        lines.append(f"Games  {self._sp_games_done}/{self._sp_total_games}  {bar}\n")
        lines.append(
            f"Examples: {self._sp_total_examples:,}    "
            f"Avg moves/game: {self._sp_avg_moves:.1f}    "
            f"Rate: {rate:.1f} games/min\n"
        )
        if self._sp_rank_net_worths:
            labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
            parts = []
            for i, v in enumerate(self._sp_rank_net_worths):
                part = f"{labels[i]}=${v:,.0f}"
                if self._sp_rank_mins and self._sp_rank_maxs:
                    part += f" [{self._sp_rank_mins[i]:,.0f}\u2013{self._sp_rank_maxs[i]:,.0f}]"
                parts.append(part)
            lines.append(f"Net worth: {', '.join(parts)}\n")
        if self._sp_games_done > 0:
            lines.append(
                f"Policy entropy: {self._sp_policy_entropy:.3f} nats    "
                f"Top-1 visit frac: {self._sp_top1_visit_frac:.1%}\n"
            )
        lines.append(f"Elapsed: {_format_duration(elapsed)}")
        return Panel(
            lines,
            title=f"Epoch {self._sp_epoch}/{self._sp_num_epochs} \u2014 Self-Play",
            border_style="blue",
        )

    # --- Live display: Training ---

    def begin_training(
        self, epoch: int, num_epochs: int, total_steps: int
    ) -> None:
        self._tr_epoch = epoch
        self._tr_num_epochs = num_epochs
        self._tr_total_steps = total_steps
        self._tr_step = 0
        self._tr_losses = {}
        self._tr_lr = 0.0
        self._phase_start = time.perf_counter()

        self._live = Live(
            self._build_training_panel(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def update_training(
        self, step: int, losses: dict[str, float], lr: float
    ) -> None:
        self._tr_step = step
        self._tr_losses = losses
        self._tr_lr = lr
        if self._live is not None:
            self._live.update(self._build_training_panel())

    def end_training(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _build_training_panel(self) -> Panel:
        elapsed = time.perf_counter() - self._phase_start
        bar = _progress_bar(self._tr_step, self._tr_total_steps)
        lines = Text()
        lines.append(f"Step  {self._tr_step}/{self._tr_total_steps}  {bar}\n")
        pl = self._tr_losses.get("policy_loss", 0.0)
        vl = self._tr_losses.get("value_loss", 0.0)
        tl = self._tr_losses.get("total_loss", 0.0)
        lines.append(f"Loss: policy={pl:.3f}  value={vl:.3f}  total={tl:.3f}\n")
        lines.append(f"LR: {self._tr_lr:.2e}    Elapsed: {_format_duration(elapsed)}")
        return Panel(
            lines,
            title=f"Epoch {self._tr_epoch}/{self._tr_num_epochs} \u2014 Training",
            border_style="green",
        )

    # --- Tensorboard ---

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)

    # --- Static summaries ---

    def log_training_start(self, config: TrainingConfig, device: str) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Players", str(config.num_players))
        table.add_row("Model", config.model_type)
        table.add_row("MCTS simulations", str(config.num_simulations))
        table.add_row("Search batch size", str(config.search_batch_size))
        table.add_row(
            "Workers",
            str(config.num_workers) if config.num_workers > 0 else "single-process",
        )
        table.add_row("Games/epoch", f"{config.games_per_epoch:,}")
        table.add_row("Training steps/epoch", f"{config.training_steps_per_epoch:,}")
        table.add_row("Batch size", str(config.batch_size))
        if config.model_type == "transformer":
            table.add_row(
                "Phase conditioning",
                "adaLN enabled" if config.phase_conditioning else "disabled",
            )
            table.add_row(
                "Price slot keys",
                f"Fourier bands={config.price_slot_fourier_bands}, "
                f"embedding blend={config.price_slot_residual_scale:g}",
            )
        else:
            table.add_row(
                "ResNet",
                f"hidden={config.resnet_hidden_dim}, "
                f"blocks={config.resnet_num_blocks}, "
                "head_layers=2",
            )
        decay_end = config.lr_decay_end_epoch or config.num_epochs
        lr_desc = (
            f"{config.learning_rate:.1e} \u2192 {config.lr_min:.1e} "
            f"(warmup {config.warmup_epochs:g} epochs, decay to epoch {decay_end})"
        )
        table.add_row("Optimizer", config.optimizer.upper())
        table.add_row("LR", lr_desc)
        table.add_row("Weight decay", f"{config.weight_decay:.1e}")
        table.add_row("Buffer capacity", f"{config.buffer_capacity:,}")
        table.add_row(
            "Temperature",
            f"{config.temp_initial} \u2192 {config.temp_final} "
            f"(anneal moves {config.temp_anneal_start}\u2013{config.temp_anneal_end})",
        )
        table.add_row(
            "c_puct",
            f"{config.c_puct_initial} \u2192 {config.c_puct_final} "
            f"(anneal {config.c_puct_anneal_epochs} epochs)",
        )
        table.add_row(
            "Value target",
            f"game outcome \u2192 A0GB "
            f"(blend epochs {config.value_blend_start_epoch}\u2013{config.value_blend_end_epoch})",
        )
        table.add_row("Epochs", str(config.num_epochs))
        table.add_row("Checkpoint interval", f"every {config.checkpoint_interval} epochs")
        table.add_row("Device", device)
        table.add_row("Tensorboard", config.tensorboard_dir)
        self.console.print(
            Panel(table, title="AlphaZero Training", border_style="bold cyan")
        )

    def log_epoch_summary(
        self,
        epoch: int,
        num_epochs: int,
        self_play_stats: dict[str, Any],
        train_stats: dict[str, float],
        buffer_size: int,
        buffer_capacity: int,
        epoch_duration: float,
        checkpoint_path: str | None = None,
    ) -> None:
        games = int(self_play_stats.get("games", 0))
        examples = int(self_play_stats.get("examples", 0))
        avg_moves = self_play_stats.get("avg_moves", 0.0)
        avg_dur = self_play_stats.get("avg_duration", 0.0)
        steps = int(train_stats.get("steps", 0))
        tl = train_stats.get("total_loss", 0.0)
        pl = train_stats.get("policy_loss", 0.0)
        vl = train_stats.get("value_loss", 0.0)
        lr = train_stats.get("lr", 0.0)
        pct = buffer_size / max(buffer_capacity, 1) * 100

        prefix = f"Epoch {epoch}/{num_epochs}"
        pad = " " * len(prefix)

        rank_nws = self_play_stats.get("rank_net_worths")
        rank_mins = self_play_stats.get("rank_net_worths_min")
        rank_maxs = self_play_stats.get("rank_net_worths_max")
        nw_str = ""
        if isinstance(rank_nws, list):
            labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
            parts = []
            for i, v in enumerate(rank_nws):
                part = f"{labels[i]}=${v:,.0f}"
                if isinstance(rank_mins, list) and isinstance(rank_maxs, list):
                    part += f" [{rank_mins[i]:,.0f}\u2013{rank_maxs[i]:,.0f}]"
                parts.append(part)
            nw_str = f"  Net worth: {', '.join(parts)}"

        self.console.print(
            f"{prefix}  Self-play: {games:,} games, {examples:,} examples "
            f"({avg_moves:.1f} moves/game, {avg_dur:.1f}s/game)"
        )
        if nw_str:
            self.console.print(f"{pad}{nw_str}")
        pol_ent = self_play_stats.get("policy_entropy", 0.0)
        top1_vf = self_play_stats.get("top1_visit_frac", 0.0)
        if games > 0:
            self.console.print(
                f"{pad}  Policy entropy: {pol_ent:.3f} nats, "
                f"top-1 visit frac: {top1_vf:.1%}"
            )
        if steps > 0:
            self.console.print(
                f"{pad}  Training: {steps:,} steps, loss={tl:.3f} "
                f"(policy={pl:.3f} value={vl:.3f}) lr={lr:.2e}"
            )
        self.console.print(
            f"{pad}  Buffer: {buffer_size:,}/{buffer_capacity:,} ({pct:.1f}%)  "
            f"Epoch time: {_format_duration(epoch_duration)}"
        )
        if checkpoint_path:
            self.console.print(f"{pad}  Checkpoint: {checkpoint_path}")

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.writer.flush()
        self.writer.close()
