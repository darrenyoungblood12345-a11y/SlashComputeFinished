"""Live view of connected nodes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

from slashcompute.common.protocol import DeviceProfile, PeerAddr, Register
from slashcompute.common.reliable import Inbox, Outbox

SendFn = Callable[[BaseModel], Awaitable[None]]
CloseFn = Callable[[], Awaitable[None]]
log = logging.getLogger(__name__)


@dataclass
class Assignment:
    job_id: str
    epoch: int
    stage_idx: int


@dataclass
class Release:
    """The node was told to stop this job's stage. It takes no new work until its agent says
    it let go: a model download can hold a Mac long after the job was cancelled."""

    job_id: str
    epoch: int
    since: float = field(default_factory=time.monotonic)
    warned: bool = False


@dataclass
class FetchState:
    """The node's agent is downloading the model for this job's stage."""

    job_id: str
    epoch: Optional[int]
    done_bytes: Optional[int]
    total_bytes: Optional[int]


@dataclass
class NodeState:
    node_id: str
    name: str
    device: DeviceProfile
    data_host: str
    data_port: int
    gpu_percent: int
    send: SendFn
    last_heartbeat: float = field(default_factory=time.monotonic)
    status: str = "idle"
    draining: bool = False
    drain_deadline: Optional[float] = None
    assignment: Optional[Assignment] = None
    verifying: Optional[str] = None  # verification id
    canary_passed: Optional[bool] = None
    last_canary: float = 0.0
    user_id: Optional[str] = None
    close: Optional[CloseFn] = None  # closes the current WebSocket
    # Reliable session (agents that send Register.session_id): messages are numbered and
    # acknowledged, and a reconnect within reconnect_grace_s keeps the node's place.
    session_id: Optional[str] = None
    outbox: Outbox = field(default_factory=Outbox)
    inbox: Inbox = field(default_factory=Inbox)
    connected: bool = True
    disconnected_at: Optional[float] = None
    # What its heartbeats say it is doing, which can lag what the coordinator decided.
    hb_job_id: Optional[str] = None
    hb_epoch: Optional[int] = None
    fetch: Optional[FetchState] = None
    releasing: Optional[Release] = None
    avoid_until: float = 0.0          # prefer other Macs until then (it stalled a start)

    @property
    def reliable(self) -> bool:
        return self.session_id is not None

    @property
    def peer(self) -> PeerAddr:
        return PeerAddr(node_id=self.node_id, host=self.data_host, port=self.data_port)

    @property
    def schedulable(self) -> bool:
        return (self.connected and not self.draining and self.assignment is None
                and self.releasing is None and self.verifying is None
                and self.canary_passed is not False)

    @property
    def in_pool(self) -> bool:
        """Could take work now or once its current work ends."""
        return not self.draining and self.canary_passed is not False

    def still_on(self, job_id: str, epoch: int) -> bool:
        """Its heartbeats say it is still busy with that job's stage."""
        return self.hb_job_id == job_id and self.hb_epoch == epoch and self.status != "idle"


class Registry:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeState] = {}

    def register(self, msg: Register, send: SendFn, close: Optional[CloseFn] = None) -> NodeState:
        state = NodeState(node_id=msg.node_id, name=msg.name, device=msg.device,
                          data_host=msg.data_host, data_port=msg.data_port,
                          gpu_percent=msg.gpu_percent, send=send, close=close,
                          session_id=msg.session_id)
        self.nodes[msg.node_id] = state
        return state

    def get(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.get(node_id)

    def remove(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.pop(node_id, None)

    def heartbeat(self, node_id: str, status: str, job_id: Optional[str] = None,
                  epoch: Optional[int] = None, *, phase: Optional[str] = None,
                  fetch_done: Optional[int] = None, fetch_total: Optional[int] = None,
                  confirm_after: float = 0.0) -> bool:
        """Record a heartbeat; True when it shows the model download moved on."""
        n = self.nodes.get(node_id)
        if n is None:
            return False
        now = time.monotonic()
        n.last_heartbeat = now
        n.status, n.hb_job_id, n.hb_epoch = status, job_id, epoch
        advanced = False
        if phase == "fetching" and job_id is not None:
            prev = n.fetch
            same = prev is not None and (prev.job_id, prev.epoch) == (job_id, epoch)
            advanced = fetch_done is not None and (
                not same or prev.done_bytes is None or fetch_done > prev.done_bytes)
            n.fetch = FetchState(job_id, epoch, fetch_done, fetch_total)
        else:
            n.fetch = None
        rel = n.releasing
        # Idle, or on another job: it let go. Only from a heartbeat sent well after we asked,
        # so one already on its way (from before it got the stage) doesn't count.
        if rel is not None and now - rel.since >= confirm_after and (
                status == "idle" or (job_id is not None and (job_id, epoch) != (rel.job_id, rel.epoch))):
            n.releasing = None
            log.info("node %s let go of job %s epoch %d", node_id[:8], rel.job_id, rel.epoch)
        return advanced

    def release(self, node: NodeState, job_id: str, epoch: int) -> None:
        """``node`` was told to stop ``job_id``/``epoch``: keep it out of new work until its
        agent confirms (``confirm_release`` or a heartbeat), or ``expire_releases`` gives up."""
        node.assignment = None
        node.releasing = Release(job_id, epoch)

    def confirm_release(self, node_id: str, job_id: str, epoch: int) -> bool:
        n = self.nodes.get(node_id)
        rel = n.releasing if n is not None else None
        if rel is None or (rel.job_id, rel.epoch) != (job_id, epoch):
            return False
        n.releasing = None
        log.info("node %s let go of job %s epoch %d", node_id[:8], job_id, epoch)
        return True

    def pool(self) -> list[NodeState]:
        return [n for n in self.nodes.values() if n.in_pool]

    def expired(self, timeout_s: float) -> list[NodeState]:
        """Connected nodes whose heartbeats stopped (disconnected ones wait out their grace)."""
        cutoff = time.monotonic() - timeout_s
        return [n for n in self.nodes.values() if n.connected and n.last_heartbeat < cutoff]

    def away(self, grace_s: float) -> list[NodeState]:
        """Disconnected nodes that did not come back within ``grace_s``."""
        cutoff = time.monotonic() - grace_s
        return [n for n in self.nodes.values()
                if not n.connected and n.disconnected_at is not None and n.disconnected_at < cutoff]

    def schedulable(self) -> list[NodeState]:
        return [n for n in self.nodes.values() if n.schedulable]
