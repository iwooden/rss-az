from __future__ import annotations

import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.token_data import TokenDataSize, get_num_tokens
from nn import create_model, get_model_input_spec
from nn.resnet import RSSResNet
from nn.transformer import RSSTransformerNet, UNIFIED_LOGIT_DIM
from train.analyze_game import analyze_game
from train.checkpoint import load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig


def _small_resnet_config() -> TrainingConfig:
    return TrainingConfig(
        model_type="resnet",
        resnet_hidden_dim=32,
        resnet_num_blocks=1,
        resnet_value_hidden_layers=0,
    )


def test_factory_instantiates_transformer_from_config() -> None:
    config = TrainingConfig(model_type="transformer")
    model = create_model(config)

    assert isinstance(model, RSSTransformerNet)
    assert model.cfg.num_players == config.num_players
    assert model.cfg.phase_conditioning is config.phase_conditioning
    assert model.cfg.price_slot_fourier_bands == config.price_slot_fourier_bands
    assert model.cfg.price_slot_residual_scale == config.price_slot_residual_scale


def test_factory_instantiates_resnet_from_config() -> None:
    config = _small_resnet_config()
    model = create_model(config)

    assert isinstance(model, RSSResNet)
    assert model.cfg.num_players == config.num_players
    assert model.cfg.hidden_dim == config.resnet_hidden_dim
    assert model.cfg.num_blocks == config.resnet_num_blocks
    assert model.cfg.value_hidden_layers == config.resnet_value_hidden_layers


def test_transformer_factory_output_shapes_remain_unified() -> None:
    config = TrainingConfig(model_type="transformer")
    model = create_model(config)
    model.eval()

    batch = 1
    num_tokens = get_num_tokens(config.num_players)
    tokens = torch.zeros(batch, num_tokens, int(TokenDataSize.TOKEN_DIM))
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


def test_resnet_factory_output_shapes_remain_unified() -> None:
    config = _small_resnet_config()
    model = create_model(config)
    assert isinstance(model, RSSResNet)
    model.eval()

    batch = 2
    x = torch.randn(batch, model.cfg.input_dim)
    legal_mask = torch.ones(batch, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)

    with torch.inference_mode():
        policy_logits, values = model(x, legal_mask)

    assert policy_logits.shape == (batch, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (batch, config.num_players)


def test_model_input_spec_marks_resnet_values_active_relative() -> None:
    transformer = get_model_input_spec(TrainingConfig(model_type="transformer"))
    resnet = get_model_input_spec(_small_resnet_config())

    assert transformer.uses_relations is True
    assert transformer.values_are_active_relative is False
    assert transformer.num_tokens == get_num_tokens(transformer.num_players)
    assert resnet.uses_relations is False
    assert resnet.values_are_active_relative is True
    assert resnet.input_dim is not None and resnet.input_dim > 0


def test_new_checkpoints_reload_correct_model_type(tmp_path) -> None:
    device = torch.device("cpu")
    cases = [
        (TrainingConfig(model_type="transformer"), RSSTransformerNet),
        (_small_resnet_config(), RSSResNet),
    ]

    for config, expected_type in cases:
        model = create_model(config).to(device)
        path = tmp_path / f"{config.model_type}.pt"

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

        assert isinstance(loaded, expected_type)
        assert loaded_config.model_type == config.model_type
        assert list(loaded.state_dict().keys()) == list(model.state_dict().keys())


def test_reloaded_resnet_checkpoint_runs_analyze_game(tmp_path) -> None:
    device = torch.device("cpu")
    config = _small_resnet_config()
    model = create_model(config).to(device)
    path = tmp_path / "resnet.pt"

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
    loaded.eval()

    rendered = analyze_game(
        loaded,
        device,
        loaded_config,
        seed=1,
        num_simulations=1,
        top_n=1,
    )

    assert "# Self-Play Analysis" in rendered
