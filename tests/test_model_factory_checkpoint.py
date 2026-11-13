from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing as mp
from multiprocessing.queues import Queue
from pathlib import Path
from typing import cast

import pytest
import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.token_data import get_num_tokens, get_token_dim
from games import get_game
from mcts.evaluator import NNEvaluator
from nn import ModelInputSpec, RSSTransformerNet, _load_model_module, create_model, get_model_input_spec
from nn.policy_layout import UNIFIED_LOGIT_DIM
from train.checkpoint import Checkpoint, load_checkpoint, load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.eval_server import SharedEvalBuffers
from train.main import _build_parser, _resolve_config

V2_PARITY_CHECKPOINT = Path(__file__).with_name("v2_parity") / "checkpoint.pt"

# A model module with its own builder, as a non-RSS game's must be.
DUMMY_MODEL_MODULE = """
import torch

INPUT_LAYOUT_VERSION = 3
calls = []


class DummyNet(torch.nn.Module):
    def __init__(self, token_dim, policy_dim):
        super().__init__()
        self.readout = torch.nn.Linear(token_dim, policy_dim)


def build_model(config, spec):
    calls.append((config, spec))
    return DummyNet(spec.token_dim, spec.policy_dim)
"""


def _put_model_class_name(model_cls: type[torch.nn.Module], queue: Queue) -> None:
    queue.put(model_cls.__name__)


def test_factory_instantiates_custom_sized_transformer() -> None:
    config = TrainingConfig(
        model_path="nn/transformer-v2.py",
        model_type="transformer",
        d_model=64,
        d_proj=16,
        num_heads=4,
        num_layers=2,
        ff_mult=1.1,
    )
    model = create_model(config)

    assert isinstance(model, RSSTransformerNet)
    assert model.cfg.d_model == 64
    assert model.cfg.d_proj == 16
    assert model.cfg.num_heads == 4
    assert model.cfg.num_layers == 2
    assert model.cfg.ff_mult == 1.1
    assert len(model.blocks) == 2
    gate = getattr(model.blocks[0], "ffn_gate")
    assert isinstance(gate, torch.nn.Linear)
    assert gate.out_features == 128


def test_factory_instantiates_mixed_transformer_at_max_player_capacity() -> None:
    config = TrainingConfig(
        model_path="nn/transformer-v2.py",
        num_players=0,
        min_players=3,
        max_players=5,
        model_type="transformer",
    )
    model = create_model(config)
    spec = get_model_input_spec(config)

    assert isinstance(model, RSSTransformerNet)
    assert model.cfg.num_players == config.effective_max_players
    assert spec.num_players == config.effective_max_players
    assert spec.value_dim == config.effective_max_players
    assert spec.num_tokens == get_num_tokens(config.effective_max_players)


def test_factory_instantiates_transformer_from_model_path() -> None:
    config = TrainingConfig(
        model_type="transformer",
        model_path="nn/transformer-v2.py",
        phase_conditioning=True,
    )
    model = create_model(config)
    cfg = getattr(model, "cfg")

    assert model.__class__.__name__ == "RSSTransformerNet"
    assert model.__class__ is RSSTransformerNet
    assert cfg.num_players == config.num_players
    assert cfg.d_model == config.d_model
    assert cfg.d_proj == config.d_proj
    assert cfg.num_heads == config.num_heads
    assert cfg.num_layers == config.num_layers
    assert cfg.ff_mult == config.ff_mult
    assert cfg.phase_conditioning is config.phase_conditioning
    assert cfg.price_slot_fourier_bands == config.price_slot_fourier_bands
    assert any(name.endswith("phase_mod.weight") for name, _ in model.named_parameters())


def test_factory_can_disable_model_path_phase_conditioning() -> None:
    config = TrainingConfig(
        model_type="transformer",
        model_path="nn/transformer-v2.py",
        phase_conditioning=False,
    )
    model = create_model(config)
    cfg = getattr(model, "cfg")

    assert cfg.phase_conditioning is False
    assert getattr(model, "phase_mod_diagnostics")() == {}
    assert all("phase_mod" not in name for name, _ in model.named_parameters())


def test_transformer_factory_output_shapes_remain_unified() -> None:
    config = TrainingConfig(model_path="nn/transformer-v2.py", model_type="transformer")
    model = create_model(config)
    model.eval()

    batch = 1
    num_tokens = get_num_tokens(config.num_players)
    tokens = torch.zeros(batch, num_tokens, get_token_dim(2))
    tokens[:, :, 0] = 1.0
    legal_mask = torch.ones(batch, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)
    relations = torch.zeros(
        batch,
        NUM_ATTENTION_RELATIONS,
        num_tokens,
        num_tokens,
        dtype=torch.uint8,
    )

    with torch.inference_mode():
        policy_logits, values = model(tokens, legal_mask, relations)

    assert policy_logits.shape == (batch, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (batch, config.num_players)


def test_model_input_spec_matches_transformer_contract() -> None:
    config = TrainingConfig(model_path="nn/transformer-v2.py", model_type="transformer")
    transformer = get_model_input_spec(config)

    assert transformer.num_tokens == get_num_tokens(transformer.num_players)


def test_new_checkpoints_reload_correct_model_type(tmp_path) -> None:
    device = torch.device("cpu")
    config = TrainingConfig(model_path="nn/transformer-v2.py", model_type="transformer")
    model = create_model(config).to(device)
    path = tmp_path / "transformer.pt"

    save_checkpoint(
        path=path,
        epoch=0,
        model=model,
        trainer_state={"global_step": 0},
        config=config,
        metrics={},
        buffer_stats={"size": 0, "capacity": 0},
    )

    loaded, loaded_config, _cp = load_model_from_checkpoint(path, device)

    assert isinstance(loaded, RSSTransformerNet)
    assert loaded_config.model_type == config.model_type
    assert list(loaded.state_dict().keys()) == list(model.state_dict().keys())


def test_checkpoint_reload_preserves_transformer_size(tmp_path) -> None:
    device = torch.device("cpu")
    config = TrainingConfig(
        model_path="nn/transformer-v2.py",
        model_type="transformer",
        d_model=64,
        d_proj=16,
        num_heads=4,
        num_layers=1,
        ff_mult=1.1,
    )
    model = create_model(config).to(device)
    path = tmp_path / "custom-transformer.pt"

    save_checkpoint(
        path=path,
        epoch=0,
        model=model,
        trainer_state={"global_step": 0},
        config=config,
        metrics={},
        buffer_stats={"size": 0, "capacity": 0},
    )

    loaded, loaded_config, _cp = load_model_from_checkpoint(path, device)

    assert isinstance(loaded, RSSTransformerNet)
    assert loaded_config.d_model == config.d_model
    assert loaded_config.d_proj == config.d_proj
    assert loaded_config.num_heads == config.num_heads
    assert loaded_config.num_layers == config.num_layers
    assert loaded_config.ff_mult == config.ff_mult
    assert loaded.cfg.d_model == config.d_model
    gate = getattr(loaded.blocks[0], "ffn_gate")
    assert isinstance(gate, torch.nn.Linear)
    assert gate.out_features == 128


def test_checkpoint_reload_uses_model_path(tmp_path) -> None:
    device = torch.device("cpu")
    config = TrainingConfig(
        model_type="transformer",
        model_path="nn/transformer-v2.py",
        phase_conditioning=False,
    )
    model = create_model(config).to(device)
    path = tmp_path / "transformer-v2.pt"

    save_checkpoint(
        path=path,
        epoch=0,
        model=model,
        trainer_state={"global_step": 0},
        config=config,
        metrics={},
        buffer_stats={"size": 0, "capacity": 0},
    )

    loaded, loaded_config, _cp = load_model_from_checkpoint(path, device)

    assert loaded_config.model_path == config.model_path
    assert loaded.__class__ is model.__class__
    assert list(loaded.state_dict().keys()) == list(model.state_dict().keys())


# A model_path file loads on first use, so spawned eval servers and workers
# import it through nn's model-path finder.
def test_model_path_class_pickles_across_spawn() -> None:
    config = TrainingConfig(
        model_type="transformer",
        model_path="nn/transformer-v2.py",
        phase_conditioning=False,
    )
    model = create_model(config)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_put_model_class_name,
        args=(model.__class__, queue),
    )

    process.start()
    process.join(timeout=30)
    if process.exitcode is None:
        process.terminate()
        process.join(timeout=5)

    assert process.exitcode == 0
    assert queue.get(timeout=1) == "RSSTransformerNet"


def test_model_module_with_build_model_builds_its_own_model(tmp_path) -> None:
    path = tmp_path / "dummy_model.py"
    path.write_text(DUMMY_MODEL_MODULE)
    module = _load_model_module(str(path))
    config = TrainingConfig(model_path=str(path))

    model = create_model(config)
    spec = get_model_input_spec(config)

    assert isinstance(model, module.DummyNet)
    assert module.calls == [(config, spec)]
    assert (spec.game, spec.layout_version, spec.token_dim) == ("rss", 3, get_token_dim(3))
    # The keyword form builds only the RSS fallback.
    with pytest.raises(TypeError, match="defines build_model"):
        create_model(num_players=3, model_path=str(path))
    # Any game reaches the builder through its own adapter, which takes only
    # its own layouts: 1889's 3 is packed, which this token module is not, and
    # it has no layout 4.
    with pytest.raises(ValueError, match="layout 3 of game '1889' is packed"):
        create_model(TrainingConfig(game="1889", model_path=str(path)))
    path_4 = tmp_path / "dummy_layout_4_model.py"
    path_4.write_text(DUMMY_MODEL_MODULE.replace("INPUT_LAYOUT_VERSION = 3", "INPUT_LAYOUT_VERSION = 4"))
    with pytest.raises(ValueError, match="Unsupported 1889 token layout version: 4"):
        create_model(TrainingConfig(game="1889", model_path=str(path_4)))
    assert len(module.calls) == 1
    path_1889 = tmp_path / "dummy_1889_model.py"
    path_1889.write_text(DUMMY_MODEL_MODULE.replace("INPUT_LAYOUT_VERSION = 3", "INPUT_LAYOUT_VERSION = 1"))
    spec_1889 = get_model_input_spec(TrainingConfig(game="1889", model_path=str(path_1889)))
    assert (spec_1889.game, spec_1889.layout_version, spec_1889.token_dim) == ("1889", 1, 108)


def test_model_module_must_declare_its_layout(tmp_path) -> None:
    path = tmp_path / "no_layout_model.py"
    path.write_text(DUMMY_MODEL_MODULE.replace("INPUT_LAYOUT_VERSION = 3\n", ""))
    config = TrainingConfig(model_path=str(path))
    message = "must define INPUT_LAYOUT_VERSION"
    with pytest.raises(ValueError, match=message):
        get_model_input_spec(config)
    with pytest.raises(ValueError, match=message):
        create_model(config)


@pytest.mark.parametrize("game", ["rss", "1889"])
def test_config_without_model_path_builds_no_model(game: str) -> None:
    config = TrainingConfig(game=game)
    assert config.model_path is None
    message = "model_path is not set"
    with pytest.raises(ValueError, match=message):
        create_model(config)
    with pytest.raises(ValueError, match=message):
        get_model_input_spec(config)
    with pytest.raises(ValueError, match=message):
        create_model(num_players=3)


@pytest.mark.parametrize("model_path", ["nn/transformer-v2.py", "nn/transformer-v3.py"])
def test_rss_transformer_fallback_refuses_other_games(model_path: str) -> None:
    config = TrainingConfig(game="1889", model_path=model_path)
    message = "game '1889' needs a model_path module that defines build_model"
    with pytest.raises(ValueError, match=message):
        create_model(config)
    with pytest.raises(ValueError, match=message):
        get_model_input_spec(config)


def test_spec_and_checkpoint_without_game_are_rss() -> None:
    """A spec or checkpoint without ``game`` is RSS's, as a config is
    (test_training_config_players.py)."""
    spec = get_model_input_spec(TrainingConfig(model_path="nn/transformer-v2.py"))
    fields = {name: getattr(spec, name) for name in (
        "model_type", "num_players", "policy_dim", "value_dim", "num_tokens", "token_dim",
        "layout_version",
    )}
    assert spec.game == "rss"
    assert ModelInputSpec(**fields) == spec

    # The v2 parity checkpoint predates the field.
    cp = load_checkpoint(V2_PARITY_CHECKPOINT, torch.device("cpu"))
    assert '"game"' not in cp["config_json"]
    _model, config, _cp = load_model_from_checkpoint(V2_PARITY_CHECKPOINT, torch.device("cpu"))
    assert config.game == "rss"
    resume = _build_parser().parse_args(["--resume", str(V2_PARITY_CHECKPOINT)])
    assert _resolve_config(resume, cp).game == "rss"


def test_evaluators_refuse_another_games_spec() -> None:
    config = TrainingConfig(model_path="nn/transformer-v2.py")
    game = get_game(config)
    spec = replace(get_model_input_spec(config), game="1889")
    message = r"input_spec game \('1889'\) does not match"
    with pytest.raises(ValueError, match=message):
        NNEvaluator(game, create_model(config), torch.device("cpu"), 3, input_spec=spec)
    with pytest.raises(ValueError, match=message):
        SharedEvalBuffers(game, num_workers=1, batch_size=1, num_players=3, input_spec=spec)


def test_resume_refuses_a_config_for_another_game(tmp_path) -> None:
    cp = cast(Checkpoint, {"config_json": TrainingConfig().to_json()})
    override = tmp_path / "1889.json"
    override.write_text(json.dumps({"game": "1889"}))
    args = _build_parser().parse_args(["--resume", "latest", "--config", str(override)])
    with pytest.raises(
        ValueError, match="the checkpoint plays game 'rss', but the run's config sets game '1889'",
    ):
        _resolve_config(args, cp)
    # Without a checkpoint, the config's game stands.
    assert _resolve_config(args, None).game == "1889"
