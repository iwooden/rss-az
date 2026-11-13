"""Neural network model loading and contract helpers.

A config's ``model_path`` names its model module, a module name or a Python
file, loaded on first use; there is no default module. Every module declares
its token layout as ``INPUT_LAYOUT_VERSION``; one whose input is a packed
vector, not token rows, also defines ``packed_input_size(max_players)``. A
module that defines
``build_model(config, spec)`` builds its model from the config and its
``ModelInputSpec``, for any game. Any other module is an RSS transformer
(``RSSTransformerNet`` and ``TransformerConfig``), the fallback, which runs
only RSS. This package's ``RSSTransformerNet``, ``TransformerBlock``, and
``TransformerConfig`` are v2's (``transformer-v2.py``), for tests and RSS
tooling.
"""

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

from games import get_game
from nn.model_contract import ModelInputSpec, ModelKind, input_spec_for, normalize_model_type


_MODEL_IMPL_MODULE_PREFIX = "_rss_model_impl_"
_MODEL_IMPL_ENV_PREFIX = "RSS_MODEL_IMPL_PATH_"
_V2_MODEL_PATH = Path(__file__).with_name("transformer-v2.py")


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


def _v2_model_module() -> ModuleType:
    """``transformer-v2.py``, loaded on first use: it imports the RSS engine."""
    return _load_model_module(str(_V2_MODEL_PATH))


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
    def __getattr__(name: str) -> Any:
        # V2's classes, loaded with its module on first use.
        if name in ("RSSTransformerNet", "TransformerBlock", "TransformerConfig"):
            return _require_symbol(_v2_model_module(), name, str(_V2_MODEL_PATH))
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _require_model_path(model_path: str | None) -> str:
    if model_path is None:
        raise ValueError("model_path is not set; name the model module, e.g. nn/transformer-v2.py")
    return model_path


def _model_module(config: object) -> tuple[ModuleType, str]:
    """The config's model module. Another game's must define ``build_model``:
    the RSS transformer fallback runs only RSS."""
    path = _require_model_path(_config_value(config, "model_path", None))
    module = _load_model_module(path)
    game = _config_value(config, "game", "rss")
    if game != "rss" and not hasattr(module, "build_model"):
        raise ValueError(
            f"game {game!r} needs a model_path module that defines build_model(config, spec): "
            f"the RSSTransformerNet fallback runs only RSS (model_path={path!r})"
        )
    return module, path


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
    module, path = _model_module(config)
    layout_version = int(_require_symbol(module, "INPUT_LAYOUT_VERSION", path))
    game = get_game(config)  # type: ignore[arg-type]
    spec = input_spec_for(game, max_players, layout_version, model_kind.value)
    # A packed module's vector is its game's layout: the game fills it.
    packed_input_size = getattr(module, "packed_input_size", None)
    module_size = 0 if packed_input_size is None else int(packed_input_size(max_players))
    if module_size != spec.input_size:
        raise ValueError(
            f"model_path {path!r} takes {module_size or 'token'} inputs a state at {max_players} players, "
            f"but layout {layout_version} of game {game.name!r} is "
            f"{f'packed, {spec.input_size} values' if spec.packed else 'a token layout'}"
        )
    return spec


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
    zero_sum_values: bool = False,
    price_slot_fourier_bands: int = 4,
    model_path: str | None = None,
) -> nn.Module:
    """Instantiate the configured model.

    With a config, a model module that defines ``build_model`` builds it from
    the config and ``get_model_input_spec(config)``. Otherwise, and in the
    keyword form (which needs ``model_path``), it is the RSS transformer
    fallback.
    """
    if isinstance(config, int):
        if num_players is not None:
            raise TypeError("Pass num_players either positionally or by keyword, not both")
        num_players = config
        config = None

    if config is None:
        if num_players is None:
            raise TypeError("create_model requires a TrainingConfig or num_players")
        selected_path = _require_model_path(model_path)
        module = _load_model_module(selected_path)
        if hasattr(module, "build_model"):
            raise TypeError(
                f"model_path {selected_path!r} defines build_model(config, spec); "
                "pass a TrainingConfig"
            )
        values: dict[str, Any] = {
            "num_players": num_players,
            "d_model": d_model,
            "d_proj": d_proj,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_mult": ff_mult,
            "phase_conditioning": phase_conditioning,
            "relation_input_mixing": relation_input_mixing,
            "zero_sum_values": zero_sum_values,
            "price_slot_fourier_bands": price_slot_fourier_bands,
        }
    else:
        normalize_model_type(
            str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
        )
        module, selected_path = _model_module(config)
        build_model = getattr(module, "build_model", None)
        if build_model is not None:
            return build_model(config, get_model_input_spec(config))
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
            "zero_sum_values": _config_value(config, "zero_sum_values", False),
            "price_slot_fourier_bands": int(
                _config_value(config, "price_slot_fourier_bands", 4)
            ),
        }

    net_cls = _require_symbol(module, "RSSTransformerNet", selected_path)
    config_cls = _require_symbol(module, "TransformerConfig", selected_path)
    return net_cls(config_cls(**_config_kwargs(config_cls, values)))
