"""A hosted pool advertises itself on the LAN so "Find on LAN" can join it."""

import logging
import re
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from zeroconf import Zeroconf

from slashcompute.common.config import MDNS_SERVICE_TYPE, EngineConfig
from slashcompute.common.discovery import discover
from slashcompute.coordinator.app import create_app

BUILD_DMG = Path(__file__).resolve().parents[1] / "scripts/macos/build_dmg.sh"


def _hosted_pool_as_found_on_the_lan(tmp_path, caplog, port: int, service: str):
    """Host a pool and resolve its advertisement by name, the way another Mac's lookup does."""
    cfg = EngineConfig(home=tmp_path / "home", coordinator_port=port)
    with caplog.at_level(logging.WARNING, logger="slashcompute.coordinator.app"):
        with TestClient(create_app(cfg, advertise=True)) as client:
            adv = client.app.state.advertiser
            assert adv is not None, caplog.text
            # Resolve this coordinator by name: another pool on the same LAN may also be advertising.
            zc = Zeroconf()
            try:
                info = zc.get_service_info(service, adv.name, timeout=3000)
            finally:
                zc.close()
    assert "mDNS advertising unavailable" not in caplog.text
    return info


def test_hosted_pool_advertises_on_the_lan(tmp_path, caplog):
    info = _hosted_pool_as_found_on_the_lan(tmp_path, caplog, 18765, MDNS_SERVICE_TYPE)
    assert info is not None and info.port == 18765
    assert info.properties.get(b"path") == b"/ws/agent"


def _classic_service() -> str:
    """The Bonjour service build_dmg.sh's header gives a build that runs beside another /compute."""
    return re.search(r"^#.*\bMDNS_SERVICE=(\S+)", BUILD_DMG.read_text(encoding="utf-8"), re.M).group(1)


def test_a_classic_pool_advertises_under_its_own_service(tmp_path, caplog, monkeypatch):
    # Bonjour allows a service name of 15 characters. zeroconf refused to register the classic
    # build's first one ("slashcompute-classic", 20), the coordinator carried on unadvertised, and
    # Find on LAN never found a classic pool.
    service = f"{_classic_service()}.local."   # as the app bundle sets SLASHCOMPUTE_MDNS_SERVICE
    assert service != MDNS_SERVICE_TYPE
    monkeypatch.setattr("slashcompute.common.discovery.MDNS_SERVICE_TYPE", service)
    info = _hosted_pool_as_found_on_the_lan(tmp_path, caplog, 18775, service)
    assert info is not None and info.port == 18775


def _build(tmp_path, service: str) -> subprocess.CompletedProcess:
    """build_dmg.sh as far as its first failing check. It never builds: this PATH has no uv."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(stage),
           "DMG_FILE": str(tmp_path / "out.dmg"), "MDNS_SERVICE": service}
    return subprocess.run(["/bin/bash", str(BUILD_DMG)], env=env, capture_output=True, text=True, timeout=60)


def test_build_refuses_a_bonjour_name_too_long_to_advertise(tmp_path):
    # Such an app would open and host, and no other Mac would ever see its pool.
    refused = _build(tmp_path, "_slashcompute-classic._tcp")
    assert refused.returncode == 1
    assert "_slashcompute-classic._tcp" in refused.stderr and "15" in refused.stderr
    assert not list((tmp_path / "stage").iterdir())   # refused before anything is staged
    for service in ("_slashcompute._tcp", _classic_service()):
        passed = _build(tmp_path, service)
        assert "MDNS_SERVICE" not in passed.stderr, passed.stderr
        assert "uv is required" in passed.stderr or "Apple Silicon" in passed.stderr   # a later check


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
