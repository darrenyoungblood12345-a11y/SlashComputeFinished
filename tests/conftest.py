import errno

import pytest

from accounts import GOOGLE_CLIENT_ID
from slashcompute.pipeline.tiny import make_tiny_dataset, make_tiny_model


@pytest.fixture(autouse=True)
def no_real_trash(monkeypatch):
    """No test may move anything to the real Trash: tests that need one patch in a folder under tmp."""
    def refuse(path):
        raise OSError(errno.EPERM, "tests may not use the real Trash")

    monkeypatch.setattr("slashcompute.inference.node.agent.move_to_trash", refuse)


@pytest.fixture(autouse=True)
def google_sign_in_configured(monkeypatch):
    """Every test's pool and app are a build with Google sign-in set up, whatever the machine
    running them has in its environment."""
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID", GOOGLE_CLIENT_ID)
    monkeypatch.delenv("SLASHCOMPUTE_GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("SLASHCOMPUTE_ADMIN_EMAIL", raising=False)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    return make_tiny_dataset(tmp_path_factory.mktemp("data") / "train.jsonl")
