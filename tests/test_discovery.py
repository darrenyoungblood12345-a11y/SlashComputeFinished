"""A hosted pool advertises itself on the LAN so "Find on LAN" can join it."""

import logging
import socket
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from zeroconf import Zeroconf

from slashcompute.common.config import MDNS_SERVICE_TYPE, EngineConfig
from slashcompute.common.discovery import discover
from slashcompute.coordinator.app import create_app


def test_hosted_pool_advertises_on_the_lan(tmp_path, caplog):
    cfg = EngineConfig(home=tmp_path / "home", coordinator_port=18765)
    with caplog.at_level(logging.WARNING, logger="slashcompute.coordinator.app"):
        with TestClient(create_app(cfg, advertise=True)) as client:
            adv = client.app.state.advertiser
            assert adv is not None, caplog.text
            # Resolve this coordinator by name: another pool on the same LAN may also be advertising.
            zc = Zeroconf()
            try:
                info = zc.get_service_info(MDNS_SERVICE_TYPE, adv.name, timeout=3000)
            finally:
                zc.close()
    assert "mDNS advertising unavailable" not in caplog.text
    assert info is not None and info.port == 18765
    assert info.properties.get(b"path") == b"/ws/agent"


def _lan(monkeypatch, advertised: dict[str, tuple[str, int] | None]) -> list:
    """A LAN where `advertised` names answer (ip, port), or nothing when None: zeroconf stood in."""
    closed = []

    class FakeZeroconf:
        def get_service_info(self, type_, name, timeout=0):
            found = advertised.get(name)
            return None if found is None else SimpleNamespace(addresses=[socket.inet_aton(found[0])], port=found[1])

        def close(self):
            closed.append(True)

    class FakeBrowser:
        def __init__(self, zc, type_, listener):
            assert type_ == MDNS_SERVICE_TYPE
            for name in advertised:
                listener.add_service(zc, type_, name)

    monkeypatch.setattr("zeroconf.Zeroconf", FakeZeroconf)
    monkeypatch.setattr("zeroconf.ServiceBrowser", FakeBrowser)
    return closed


def test_discover_skips_what_the_caller_excludes_and_takes_the_next(monkeypatch):
    # While hosting, the first advertisement to answer is this Mac's own pool: discover() used to
    # return it and stop, so "Find on LAN" never found another pool.
    closed = _lan(monkeypatch, {
        f"slashcompute-mine.{MDNS_SERVICE_TYPE}": ("192.168.1.20", 8765),
        f"slashcompute-ghost.{MDNS_SERVICE_TYPE}": None,                    # never resolves
        f"slashcompute-theirs.{MDNS_SERVICE_TYPE}": ("192.168.1.21", 8765),
    })
    assert discover(1.0) == "http://192.168.1.20:8765"                       # the agent's fast path
    assert discover(1.0, exclude=lambda url: url.startswith("http://192.168.1.20:")) == "http://192.168.1.21:8765"
    began = time.monotonic()
    assert discover(0.2, exclude=lambda url: True) is None                   # only ours: give up at the timeout
    assert 0.15 < time.monotonic() - began < 1.0
    assert len(closed) == 3
