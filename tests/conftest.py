import errno

import pytest

from slashcompute.pipeline.tiny import make_tiny_dataset, make_tiny_model


@pytest.fixture(autouse=True)
def no_real_trash(monkeypatch):
    """No test may move anything to the real Trash: tests that need one patch in a folder under tmp."""
    def refuse(path):
        raise OSError(errno.EPERM, "tests may not use the real Trash")

    monkeypatch.setattr("slashcompute.inference.node.agent.move_to_trash", refuse)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    return make_tiny_dataset(tmp_path_factory.mktemp("data") / "train.jsonl")
