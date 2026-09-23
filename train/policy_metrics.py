"""Decision-weighted diagnostics comparing unnoised priors, search and targets.

Only real decisions with at least two available actions enter the averages.
These are search disagreements, not demonstrated improvements in playing strength.
"""

from dataclasses import dataclass, field

import numpy as np


# Matches core.data.DecisionPhase; shared with trainer reporting.
PHASE_NAMES = (
    "invest", "bid", "acq_corp", "acq_offer", "close", "div", "issue",
    "ipo", "par", "acq_co", "acq_price",
)

_COUNTS = (
    "decision_count", "forced_count", "search_changed_count",
    "prior_above_95_count", "prior_above_98_count",
)
_MEANS = (
    "prior_top1_mean", "prior_entropy_mean",
    "search_top1_mean", "search_entropy_mean", "search_unvisited_fraction_mean",
    "target_top1_mean", "target_entropy_mean", "search_choice_prior_mean",
    "search_kl_to_prior_mean", "target_kl_to_prior_mean",
    "target_kl_different_top_contribution",
    "target_kl_same_top_prior_sharper_contribution",
    "target_kl_same_top_prior_not_sharper_contribution",
    "same_top_prior_sharper_fraction",
)
_CONDITIONAL = {
    "search_changed_choice_prior_mean": "search_changed_count",
    "search_changed_from_above_95_fraction": "prior_above_95_count",
    "search_changed_from_above_98_fraction": "prior_above_98_count",
}
_KEYS = (*_COUNTS, *_MEANS, *_CONDITIONAL)
_INDEX = {key: i for i, key in enumerate(_KEYS)}


@dataclass
class PolicyMetrics:
    """Small additive payload for worker IPC and epoch aggregation.

    Stages refer to the configured target-temperature window, not game turns.
    Priors are normalized over the actions allowed by the search policy (including
    the acquisition-price restriction), captured before root Dirichlet noise.
    """

    sums: dict[str, np.ndarray] = field(default_factory=dict)

    def observe(
        self,
        priors: np.ndarray,
        visits: np.ndarray,
        target: np.ndarray,
        phase_id: int,
        move_count: int,
        anneal_window: tuple[int, int],
    ) -> None:
        start, end = anneal_window
        stage = (
            "pre_anneal" if move_count <= start
            else "post_anneal" if move_count >= end else "anneal"
        )
        row = np.zeros(len(_KEYS), dtype=np.float64)
        if len(priors) < 2:
            row[_INDEX["forced_count"]] = 1
        else:
            # Float64 keeps diagnostics stable; none of these arrays feed search.
            p = priors.astype(np.float64)
            p /= p.sum()
            r = visits.astype(np.float64)
            r /= r.sum()
            q = target.astype(np.float64)
            q /= q.sum()
            distributions = np.stack((p, r, q))
            # Finite diagnostics even for float32 softmax underflow. Zero-mass
            # target/visit terms contribute zero, including unvisited actions.
            logs = np.log(np.maximum(distributions, 1e-12))
            entropy = -(distributions * logs).sum(axis=1)
            p_top = float(p.max())
            # Resolve tied visit winners in favor of the largest prior. A
            # search/prior tie is agreement, not evidence of overturning it.
            winners = np.flatnonzero(visits == visits.max())
            choice = int(winners[np.argmax(p[winners])])
            changed = bool(p[choice] < p_top - 1e-7)
            high95, high98 = p_top > 0.95, p_top > 0.98
            target_changed = bool(p[q == q.max()].max() < p_top - 1e-7)
            sharper = not target_changed and entropy[0] < entropy[2] - 1e-7
            search_kl = max(0.0, float((r * (logs[1] - logs[0])).sum()))
            target_kl = max(0.0, float((q * (logs[2] - logs[0])).sum()))
            values = {
                "decision_count": 1,
                "search_changed_count": changed,
                "prior_above_95_count": high95,
                "prior_above_98_count": high98,
                "prior_top1_mean": p_top,
                "prior_entropy_mean": entropy[0],
                "search_top1_mean": r.max(),
                "search_entropy_mean": entropy[1],
                "search_unvisited_fraction_mean": np.mean(visits == 0),
                "target_top1_mean": q.max(),
                "target_entropy_mean": entropy[2],
                "search_choice_prior_mean": p[choice],
                "search_kl_to_prior_mean": search_kl,
                "target_kl_to_prior_mean": target_kl,
                "target_kl_different_top_contribution": target_kl if target_changed else 0,
                "target_kl_same_top_prior_sharper_contribution": target_kl if sharper else 0,
                "target_kl_same_top_prior_not_sharper_contribution": (
                    target_kl if not target_changed and not sharper else 0
                ),
                "same_top_prior_sharper_fraction": sharper,
                "search_changed_choice_prior_mean": p[choice] if changed else 0,
                "search_changed_from_above_95_fraction": changed and high95,
                "search_changed_from_above_98_fraction": changed and high98,
            }
            for key, value in values.items():
                row[_INDEX[key]] = value
        for group in ("all", f"phase/{PHASE_NAMES[phase_id]}", f"stage/{stage}"):
            if group not in self.sums:
                self.sums[group] = row.copy()
            else:
                self.sums[group] += row

    def add(self, other: "PolicyMetrics") -> None:
        for group, row in other.sums.items():
            if group not in self.sums:
                self.sums[group] = row.copy()
            else:
                self.sums[group] += row

    def scalars(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for group, row in self.sums.items():
            prefix = f"policy/{group}"
            for key in _COUNTS:
                result[f"{prefix}/{key}"] = float(row[_INDEX[key]])
            count = row[_INDEX["decision_count"]]
            if not count:
                continue
            for key in _MEANS:
                result[f"{prefix}/{key}"] = float(row[_INDEX[key]] / count)
            for name, numerator in (
                ("prior_above_95_fraction", "prior_above_95_count"),
                ("prior_above_98_fraction", "prior_above_98_count"),
                ("search_top1_changed_fraction", "search_changed_count"),
            ):
                result[f"{prefix}/{name}"] = float(row[_INDEX[numerator]] / count)
            for key, denominator in _CONDITIONAL.items():
                denom = row[_INDEX[denominator]]
                # Missing conditional populations aren't zero measurements.
                if denom:
                    result[f"{prefix}/{key}"] = float(row[_INDEX[key]] / denom)
        return result
