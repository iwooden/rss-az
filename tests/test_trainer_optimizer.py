"""Lock in Trainer optimizer param routing.

Regressions this catches:
- norm scale weights (LayerNorm/RMSNorm) silently routed to a decay group
- bias params routed to a decay group
- embedding/anchor tables routed to a decay group
- relation attention-bias multipliers routed to a decay group
- phase-conditioning modulation weights routed to a decay group
- Muon claiming embedding/anchor tables instead of leaving them to AdamW
- any trainable param orphaned from, or double-claimed by, the optimizer(s)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nn.transformer import RSSTransformerNet, TransformerConfig
from train.config import TrainingConfig
from train.trainer import Trainer

NUM_PLAYERS = 3

_NORM_TYPES: tuple[type[nn.Module], ...] = (nn.LayerNorm, nn.RMSNorm)


def _make_trainer(optimizer: str) -> Trainer:
    model = RSSTransformerNet(
        TransformerConfig(num_players=NUM_PLAYERS, phase_conditioning=True)
    )
    cfg = TrainingConfig(
        num_players=NUM_PLAYERS,
        optimizer=optimizer,
        learning_rate=3e-4,
        weight_decay=0.01,
        batch_size=8,
        num_epochs=1,
        training_steps_per_epoch=10,
        warmup_epochs=0,
    )
    return Trainer(model, cfg, torch.device("cpu"))


def _all_groups(trainer: Trainer) -> list[dict[str, object]]:
    groups = list(trainer.optimizer.param_groups)
    if trainer._aux_optimizer is not None:
        groups.extend(trainer._aux_optimizer.param_groups)
    return groups


def _group_of(
    pid: int, groups: list[dict[str, object]]
) -> dict[str, object] | None:
    for g in groups:
        for p in g["params"]:  # type: ignore[attr-defined]
            if id(p) == pid:
                return g
    return None


def _norm_param_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, _NORM_TYPES):
            for p in module.parameters(recurse=False):
                ids.add(id(p))
    return ids


def _bias_param_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if pname == "bias":
                ids.add(id(p))
    return ids


def _embedding_param_ids(model: nn.Module) -> dict[str, int]:
    ids: dict[str, int] = {}
    for module_name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{pname}" if module_name else pname
            if isinstance(module, nn.Embedding) or pname.endswith("embeds"):
                ids[full_name] = id(p)
    return ids


def _relation_bias_param_ids(model: nn.Module) -> dict[str, int]:
    ids: dict[str, int] = {}
    for module_name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            if pname == "relation_bias_mult":
                full_name = f"{module_name}.{pname}" if module_name else pname
                ids[full_name] = id(p)
    return ids


def _phase_mod_param_ids(model: nn.Module) -> dict[str, int]:
    ids: dict[str, int] = {}
    for module_name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            if module_name.endswith("phase_mod") and pname == "weight":
                full_name = f"{module_name}.{pname}" if module_name else pname
                ids[full_name] = id(p)
    return ids


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_every_trainable_param_claimed_exactly_once(optimizer: str) -> None:
    trainer = _make_trainer(optimizer)
    model_ids = {id(p) for p in trainer.model.parameters() if p.requires_grad}

    seen: list[int] = []
    for g in _all_groups(trainer):
        for p in g["params"]:  # type: ignore[attr-defined]
            seen.append(id(p))

    assert len(seen) == len(set(seen)), "param claimed by multiple groups"
    assert set(seen) == model_ids, "optimizer param set != model param set"


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_norm_weights_are_not_decayed(optimizer: str) -> None:
    trainer = _make_trainer(optimizer)
    groups = _all_groups(trainer)
    norm_ids = _norm_param_ids(trainer.model)
    # Sanity: the model has norm weights at all.
    assert norm_ids, "expected model to contain LayerNorm/RMSNorm params"

    for pid in norm_ids:
        g = _group_of(pid, groups)
        assert g is not None, "norm param not routed to any group"
        assert g["weight_decay"] == 0.0, (
            f"norm weight routed to weight_decay={g['weight_decay']} group"
        )


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_bias_params_are_not_decayed(optimizer: str) -> None:
    trainer = _make_trainer(optimizer)
    groups = _all_groups(trainer)
    bias_ids = _bias_param_ids(trainer.model)
    assert bias_ids, "expected model to contain bias params"

    for pid in bias_ids:
        g = _group_of(pid, groups)
        assert g is not None, "bias param not routed to any group"
        assert g["weight_decay"] == 0.0, (
            f"bias routed to weight_decay={g['weight_decay']} group"
        )


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_embedding_params_are_not_decayed(optimizer: str) -> None:
    trainer = _make_trainer(optimizer)
    groups = _all_groups(trainer)
    embedding_ids = _embedding_param_ids(trainer.model)
    assert embedding_ids, "expected model to contain embedding params"

    for name, pid in embedding_ids.items():
        g = _group_of(pid, groups)
        assert g is not None, f"{name} not routed to any optimizer group"
        assert g["weight_decay"] == 0.0, (
            f"{name} routed to weight_decay={g['weight_decay']} group"
        )


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_relation_bias_params_are_not_decayed(optimizer: str) -> None:
    trainer = _make_trainer(optimizer)
    groups = _all_groups(trainer)
    relation_bias_ids = _relation_bias_param_ids(trainer.model)
    assert relation_bias_ids, "expected model to contain relation bias params"

    for name, pid in relation_bias_ids.items():
        g = _group_of(pid, groups)
        assert g is not None, f"{name} not routed to any optimizer group"
        assert g["weight_decay"] == 0.0, (
            f"{name} routed to weight_decay={g['weight_decay']} group"
        )


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_phase_mod_params_are_not_decayed(optimizer: str) -> None:
    trainer = _make_trainer(optimizer)
    groups = _all_groups(trainer)
    phase_mod_ids = _phase_mod_param_ids(trainer.model)
    assert phase_mod_ids, "expected model to contain phase_mod params"

    for name, pid in phase_mod_ids.items():
        g = _group_of(pid, groups)
        assert g is not None, f"{name} not routed to any optimizer group"
        assert g["weight_decay"] == 0.0, (
            f"{name} routed to weight_decay={g['weight_decay']} group"
        )


def test_muon_routes_phase_mod_params_to_aux_adamw_no_decay_group() -> None:
    trainer = _make_trainer("muon")
    assert trainer._aux_optimizer is not None, "expected Muon to have aux AdamW"

    muon_ids = {
        id(p)
        for g in trainer.optimizer.param_groups
        for p in g["params"]
    }
    aux_groups = list(trainer._aux_optimizer.param_groups)
    phase_mod_ids = _phase_mod_param_ids(trainer.model)
    assert phase_mod_ids, "expected model to contain phase_mod params"

    for name, pid in phase_mod_ids.items():
        assert pid not in muon_ids, f"{name} was routed to Muon"
        g = _group_of(pid, aux_groups)
        assert g is not None, f"{name} not routed to aux AdamW"
        assert g["weight_decay"] == 0.0, (
            f"{name} routed to aux AdamW with weight_decay={g['weight_decay']}"
        )


def test_muon_routes_embedding_params_to_aux_adamw_no_decay_group() -> None:
    trainer = _make_trainer("muon")
    assert trainer._aux_optimizer is not None, "expected Muon to have aux AdamW"

    muon_ids = {
        id(p)
        for g in trainer.optimizer.param_groups
        for p in g["params"]
    }
    aux_groups = list(trainer._aux_optimizer.param_groups)
    embedding_ids = _embedding_param_ids(trainer.model)
    assert embedding_ids, "expected model to contain embedding params"

    for name, pid in embedding_ids.items():
        assert pid not in muon_ids, f"{name} was routed to Muon"
        g = _group_of(pid, aux_groups)
        assert g is not None, f"{name} not routed to aux AdamW"
        assert g["weight_decay"] == 0.0, (
            f"{name} routed to aux AdamW with weight_decay={g['weight_decay']}"
        )
