"""Neural network model loading and contract helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from core.token_data import get_token_dim, get_num_tokens
from nn.model_contract import ModelInputSpec, ModelKind, normalize_model_type
from nn.policy_layout import UNIFIED_LOGIT_DIM


_MODEL_IMPL_MODULE_PREFIX = "_rss_model_impl_"
_MODEL_IMPL_ENV_PREFIX = "RSS_MODEL_IMPL_PATH_"
_DEFAULT_MODEL_PATH = Path(__file__).with_name("transformer-v2.py")


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


def _require_symbol(module: ModuleType, name: str, model_path: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ValueError(f"model_path {model_path!r} must define {name}") from exc


_ensure_model_path_finder()
_default_model_module = _load_model_module(str(_DEFAULT_MODEL_PATH))
if TYPE_CHECKING:
    # The implementation filename is intentionally file-loadable rather than
    # importable as a Python module. Its source is still checked directly;
    # these declarations describe the open implementation surface exported by
    # this loader without duplicating v2's model definition.
    class TransformerConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def __getattr__(self, name: str) -> Any: ...

    class TransformerBlock(nn.Module):
        def __getattr__(self, name: str) -> Any: ...

    class RSSTransformerNet(nn.Module):
        cfg: TransformerConfig

        def __init__(self, cfg: TransformerConfig) -> None: ...
        def __getattr__(self, name: str) -> Any: ...
else:
    RSSTransformerNet = _require_symbol(
        _default_model_module, "RSSTransformerNet", str(_DEFAULT_MODEL_PATH)
    )
    TransformerBlock = _require_symbol(
        _default_model_module, "TransformerBlock", str(_DEFAULT_MODEL_PATH)
    )
    TransformerConfig = _require_symbol(
        _default_model_module, "TransformerConfig", str(_DEFAULT_MODEL_PATH)
    )


__all__ = [
    "ModelInputSpec",
    "ModelKind",
    "RSSTransformerNet",
    "TransformerBlock",
    "TransformerConfig",
    "create_model",
    "get_model_input_spec",
]


def _config_value(config: object, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _effective_max_players(config: object) -> int:
    """Return the model/storage player capacity for transformer inputs."""
    value = _config_value(config, "effective_max_players", None)
    if value is not None:
        return int(value)
    return int(_config_value(config, "num_players", 3))


def _model_module(config: object) -> tuple[ModuleType, str]:
    model_path = _config_value(config, "model_path", None)
    if model_path is None:
        return _default_model_module, str(_DEFAULT_MODEL_PATH)
    path = str(model_path)
    return _load_model_module(path), path


def _config_kwargs(config_cls: type[object], values: dict[str, Any]) -> dict[str, Any]:
    """Filter unified TrainingConfig-derived values to a model config schema."""
    if is_dataclass(config_cls):
        valid = {field.name for field in fields(config_cls) if field.init}
        return {key: value for key, value in values.items() if key in valid}

    signature = inspect.signature(config_cls)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def get_model_input_spec(config: object) -> ModelInputSpec:
    """Resolve the active model's input/output dimensions."""
    model_kind = normalize_model_type(
        str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
    )
    max_players = _effective_max_players(config)
    module, _ = _model_module(config)
    layout_version = int(getattr(module, "INPUT_LAYOUT_VERSION", 2))
    return ModelInputSpec(
        model_type=model_kind.value,
        num_players=max_players,
        policy_dim=int(UNIFIED_LOGIT_DIM),
        value_dim=max_players,
        num_tokens=get_num_tokens(max_players),
        token_dim=get_token_dim(layout_version),
        layout_version=layout_version,
    )


def create_model(
    config: object | int | None = None,
    *,
    num_players: int | None = None,
    d_model: int = 256,
    d_proj: int = 64,
    num_heads: int = 4,
    num_layers: int = 15,
    ff_mult: float = 3.0,
    phase_conditioning: bool = False,
    relation_input_mixing: bool = True,
    price_slot_fourier_bands: int = 4,
    model_path: str | None = None,
) -> nn.Module:
    """Instantiate the configured transformer implementation."""
    if isinstance(config, int):
        if num_players is not None:
            raise TypeError("Pass num_players either positionally or by keyword, not both")
        num_players = config
        config = None

    if config is None:
        if num_players is None:
            raise TypeError("create_model requires a TrainingConfig or num_players")
        selected_path = model_path or str(_DEFAULT_MODEL_PATH)
        module = _load_model_module(selected_path)
        values: dict[str, Any] = {
            "num_players": num_players,
            "d_model": d_model,
            "d_proj": d_proj,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_mult": ff_mult,
            "phase_conditioning": phase_conditioning,
            "relation_input_mixing": relation_input_mixing,
            "price_slot_fourier_bands": price_slot_fourier_bands,
        }
    else:
        normalize_model_type(
            str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
        )
        module, selected_path = _model_module(config)
        values = {
            "num_players": _effective_max_players(config),
            "d_model": int(_config_value(config, "d_model", 256)),
            "d_proj": int(_config_value(config, "d_proj", 64)),
            "num_heads": int(_config_value(config, "num_heads", 4)),
            "num_layers": int(_config_value(config, "num_layers", 15)),
            "ff_mult": float(_config_value(config, "ff_mult", 3.0)),
            "phase_conditioning": bool(
                _config_value(config, "phase_conditioning", False)
            ),
            "relation_input_mixing": _config_value(config, "relation_input_mixing", True),
            "price_slot_fourier_bands": int(
                _config_value(config, "price_slot_fourier_bands", 4)
            ),
        }

    net_cls = _require_symbol(module, "RSSTransformerNet", selected_path)
    config_cls = _require_symbol(module, "TransformerConfig", selected_path)
    return net_cls(config_cls(**_config_kwargs(config_cls, values)))
