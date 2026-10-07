"""Engine-wide defaults. Every value can be overridden with an environment
variable named ``SLASHCOMPUTE_<FIELD>`` (upper-case)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

DEV_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Ordered smallest to largest. The planner picks the largest one that fits the
# pool but not any single node.
DEMO_MODEL_CANDIDATES = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-32B-Instruct-4bit",
]

ALLOWED_MODELS = (DEV_MODEL, *DEMO_MODEL_CANDIDATES)

MDNS_SERVICE_TYPE = "_slashcompute._tcp.local."

log = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
# Numeric fields default to a floor of 0; these need more.
_MIN = {"checkpoint_every": 1}


# One-time credit for every account on first sign-in: 1 PFLOP (credits are 1:1 FLOPs).
WELCOME_FLOPS = 1e15


def allowed_model(model: str) -> bool:
    """Catalog HF ids, or a local directory the coordinator already has."""
    name = (model or "").strip()
    if name in ALLOWED_MODELS:
        return True
    return Path(name).expanduser().is_dir()


@dataclass
class EngineConfig:
    home: Path = field(default_factory=lambda: Path.home() / ".slashcompute")
    coordinator_host: str = "0.0.0.0"
    coordinator_port: int = 8765
    data_port: int = 9700

    checkpoint_every: int = 25
    grace_period_s: float = 60.0
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 20.0
    scheduler_tick_s: float = 1.0
    # A starting epoch aborts after this long without progress: model download bytes, a
    # stage becoming ready. (Not since the assignment: a big model takes longer to download.)
    stage_start_timeout_s: float = 900.0
    # After a start timed out, prefer other Macs for this long (that one made no progress).
    start_timeout_avoid_s: float = 600.0
    # A node told to stop a job takes no new work until its agent confirms; give up waiting
    # after this long unless its heartbeats still name the job.
    release_timeout_s: float = 60.0
    max_recoveries: int = 20
    # Longest a pipeline neighbour may stay silent (and a dropped peer link may take to
    # reconnect) before the stage gives up. Generous: legitimate gaps include checkpoint
    # uploads and pacing at a low gpu_percent.
    peer_timeout_s: float = 600.0
    # The coordinator aborts a running epoch that has reported no step for this long.
    stall_timeout_s: float = 1800.0
    # A dropped agent that reconnects within this window keeps its stage, and the messages
    # either side sent meanwhile are replayed. Longer than the agent's reconnect backoff.
    reconnect_grace_s: float = 60.0
    canary_size: int = 512
    canary_interval_s: float = 1800.0

    verify_rate: float = 0.05
    verify_rel_tolerance: float = 2e-2
    # GPU fp32 vs NumPy reference; zeros/random still fail (~1.0).
    canary_rel_tolerance: float = 0.5
    verify_ring_size: int = 8

    # Fraction of each stage's weight bytes reserved for activations, grads and
    # optimizer state, plus a fixed headroom per stage.
    stage_overhead_frac: float = 0.25
    stage_overhead_bytes: int = 512 * 1024**2

    sandbox: bool = True

    # Set on a hosted coordinator. Local LAN host leaves these alone.
    public_pool: bool = False
    public_url: str = ""

    # SLASHCOMPUTE_WELCOME_FLOPS=0 disables the sign-in credit.
    welcome_flops: float = WELCOME_FLOPS

    @classmethod
    def from_env(cls, **overrides) -> "EngineConfig":
        cfg = cls()
        for f in fields(cls):
            name = f"SLASHCOMPUTE_{f.name.upper()}"
            env = os.environ.get(name)
            if env is None:
                continue
            val = _parse_env(env.strip(), getattr(cfg, f.name), _MIN.get(f.name, 0))
            if val is None:
                log.warning("ignoring %s=%r; using default %r", name, env, getattr(cfg, f.name))
                continue
            setattr(cfg, f.name, val)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    @property
    def coordinator_dir(self) -> Path:
        return self.home / "coordinator"

    @property
    def agent_dir(self) -> Path:
        return self.home / "agent"


def _parse_env(env: str, cur, minimum):
    """Parse ``env`` like ``cur``; None means unusable, keep the default."""
    if isinstance(cur, bool):
        low = env.lower()
        return True if low in _TRUE else False if low in _FALSE else None
    if isinstance(cur, Path):
        return Path(env).expanduser() if env else None
    if isinstance(cur, str):
        return env
    try:
        # int(float()) so "5e8" and "25.0" work for byte counts and step counts.
        val = int(float(env)) if isinstance(cur, int) else float(env)
    except (ValueError, OverflowError):
        return None
    return val if val >= minimum else None
