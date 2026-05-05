"""Neural network models for Rolling Stock Stars AlphaZero training."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch.nn as nn

from core.resnet_data import get_resnet_vector_size
from core.token_data import TokenDataSize, get_num_tokens
from nn.model_contract import (
    ModelInputSpec,
    ModelKind,
    canonical_player_for_relative,
    normalize_model_type,
    relative_slot_for_canonical,
    rotate_values_to_relative,
    unrotate_values_to_canonical,
)
from nn.resnet import RSSResNet, RSSResNetConfig
from nn.transformer import RSSTransformerNet, TransformerConfig, UNIFIED_LOGIT_DIM

_MODEL_IMPL_MODULE_PREFIX = "_rss_model_impl_"
_MODEL_IMPL_ENV_PREFIX = "RSS_MODEL_IMPL_PATH_"


class _ModelPathFinder(importlib.abc.MetaPathFinder):
    """Let spawned child processes import file-backed model implementations."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_MODEL_IMPL_MODULE_PREFIX):
            return None
        digest = fullname.removeprefix(_MODEL_IMPL_MODULE_PREFIX)
        model_path = os.environ.get(f"{_MODEL_IMPL_ENV_PREFIX}{digest}")
        if not model_path:
            return None
        path_obj = Path(model_path)
        if not path_obj.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, path_obj)


def _ensure_model_path_finder() -> None:
    if not any(isinstance(finder, _ModelPathFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ModelPathFinder())


_ensure_model_path_finder()

__all__ = [
    "ModelInputSpec",
    "ModelKind",
    "RSSResNet",
    "RSSResNetConfig",
    "RSSTransformerNet",
    "TransformerConfig",
    "canonical_player_for_relative",
    "create_model",
    "get_model_input_spec",
    "relative_slot_for_canonical",
    "rotate_values_to_relative",
    "unrotate_values_to_canonical",
]


def _config_value(config: object, name: str, default: object) -> object:
    return getattr(config, name, default)


def _load_model_module(model_path: str) -> ModuleType:
    """Load a model implementation from a module name or Python file path."""
    looks_like_file = model_path.endswith(".py") or "/" in model_path or "\\" in model_path
    if looks_like_file:
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise ValueError(f"model_path file does not exist: {model_path!r}")
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        os.environ[f"{_MODEL_IMPL_ENV_PREFIX}{digest}"] = str(path)
        module_name = f"{_MODEL_IMPL_MODULE_PREFIX}{digest}"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load model_path file: {model_path!r}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(model_path)


def _model_module(config: object) -> ModuleType | None:
    model_path = _config_value(config, "model_path", None)
    if model_path is None:
        return None
    return _load_model_module(str(model_path))


def _resnet_input_dim(num_players: int) -> int:
    return get_resnet_vector_size(num_players)


def _config_kwargs(config_cls: type[object], values: dict[str, object]) -> dict[str, object]:
    """Filter unified TrainingConfig-derived values to a model config schema."""
    if is_dataclass(config_cls):
        valid = {
            field.name
            for field in fields(config_cls)
            if field.init
        }
        return {k: v for k, v in values.items() if k in valid}

    # Fallback for config classes that are not dataclasses.
    import inspect

    signature = inspect.signature(config_cls)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return values
    return {k: v for k, v in values.items() if k in signature.parameters}


def _require_symbol(module: ModuleType, name: str, model_path: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ValueError(
            f"model_path {model_path!r} must define {name}"
        ) from exc


def get_model_input_spec(config: object) -> ModelInputSpec:
    """Resolve model input/output dimensions from a training config."""
    num_players = int(_config_value(config, "num_players", 3))
    model_kind = normalize_model_type(
        str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
    )

    if model_kind is ModelKind.TRANSFORMER:
        return ModelInputSpec(
            model_type=model_kind.value,
            num_players=num_players,
            policy_dim=int(UNIFIED_LOGIT_DIM),
            value_dim=num_players,
            input_dim=None,
            num_tokens=get_num_tokens(num_players),
            token_dim=int(TokenDataSize.TOKEN_DIM),
            uses_relations=True,
            values_are_active_relative=False,
        )

    return ModelInputSpec(
        model_type=model_kind.value,
        num_players=num_players,
        policy_dim=int(UNIFIED_LOGIT_DIM),
        value_dim=num_players,
        input_dim=_resnet_input_dim(num_players),
        num_tokens=None,
        token_dim=None,
        uses_relations=False,
        values_are_active_relative=True,
    )


def create_model(
    config: object | int | None = None,
    *,
    num_players: int | None = None,
    phase_conditioning: bool = False,
    price_slot_fourier_bands: int = 4,
    price_slot_residual_scale: float = 1.0,
    model_path: str | None = None,
) -> nn.Module:
    """Instantiate the configured model family.

    Preferred usage is ``create_model(training_config)``. The legacy
    ``create_model(num_players=..., ...)`` path is retained for targeted tests
    and utility callers that still construct the transformer directly.
    """
    if isinstance(config, int):
        if num_players is not None:
            raise TypeError("Pass num_players either positionally or by keyword, not both")
        num_players = config
        config = None

    if config is None:
        if num_players is None:
            raise TypeError("create_model requires a TrainingConfig or num_players")
        if model_path is not None:
            module = _load_model_module(model_path)
            net_cls = _require_symbol(module, "RSSTransformerNet", model_path)
            config_cls = _require_symbol(module, "TransformerConfig", model_path)
        else:
            net_cls = RSSTransformerNet
            config_cls = TransformerConfig
        values: dict[str, object] = {
            "num_players": num_players,
            "phase_conditioning": phase_conditioning,
            "price_slot_fourier_bands": price_slot_fourier_bands,
            "price_slot_residual_scale": price_slot_residual_scale,
        }
        return net_cls(config_cls(**_config_kwargs(config_cls, values)))

    cfg_num_players = int(_config_value(config, "num_players", 3))
    model_kind = normalize_model_type(
        str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
    )
    module = _model_module(config)
    cfg_model_path = str(_config_value(config, "model_path", "")) if module is not None else ""

    if model_kind is ModelKind.TRANSFORMER:
        net_cls = (
            _require_symbol(module, "RSSTransformerNet", cfg_model_path)
            if module is not None else RSSTransformerNet
        )
        config_cls = (
            _require_symbol(module, "TransformerConfig", cfg_model_path)
            if module is not None else TransformerConfig
        )
        values = {
            "num_players": cfg_num_players,
            "phase_conditioning": bool(
                _config_value(config, "phase_conditioning", False)
            ),
            "price_slot_fourier_bands": int(
                _config_value(config, "price_slot_fourier_bands", 4)
            ),
            "price_slot_residual_scale": float(
                _config_value(config, "price_slot_residual_scale", 1.0)
            ),
        }
        return net_cls(config_cls(**_config_kwargs(config_cls, values)))

    net_cls = (
        _require_symbol(module, "RSSResNet", cfg_model_path)
        if module is not None else RSSResNet
    )
    config_cls = (
        _require_symbol(module, "RSSResNetConfig", cfg_model_path)
        if module is not None else RSSResNetConfig
    )
    values = {
        "num_players": cfg_num_players,
        "input_dim": _resnet_input_dim(cfg_num_players),
        "hidden_dim": int(_config_value(config, "resnet_hidden_dim", 256)),
        "num_blocks": int(_config_value(config, "resnet_num_blocks", 8)),
    }
    return net_cls(config_cls(**_config_kwargs(config_cls, values)))
