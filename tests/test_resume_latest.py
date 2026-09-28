from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.checkpoint import find_latest_checkpoint
from train.main import _build_parser, _latest_checkpoint_dir


def test_resume_latest_searches_cli_then_config_then_default_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for directory, epochs in (("checkpoints", (195,)), ("run", (5, 10)), ("cli", (1,))):
        Path(directory).mkdir()
        for epoch in epochs:
            (Path(directory) / f"checkpoint_epoch_{epoch:04d}.pt").touch()
    Path("config.json").write_text(json.dumps({"checkpoint_dir": "run"}))

    def latest(*argv: str) -> Path | None:
        args = _build_parser().parse_args(["--resume", "latest", *argv])
        return find_latest_checkpoint(_latest_checkpoint_dir(args))

    assert latest() == Path("checkpoints/checkpoint_epoch_0195.pt")
    assert latest("--config", "config.json") == Path("run/checkpoint_epoch_0010.pt")
    assert latest("--config", "config.json", "--checkpoint-dir", "cli") == Path(
        "cli/checkpoint_epoch_0001.pt",
    )
    # A config without checkpoint_dir falls back to the default directory.
    Path("config.json").write_text("{}")
    assert latest("--config", "config.json") == Path("checkpoints/checkpoint_epoch_0195.pt")
    with pytest.raises(FileNotFoundError):
        latest("--config", "missing.json")
