"""Control-plane messages exchanged over the agent <-> coordinator WebSocket
and the daemon <-> worker stdio pipe. Every message is a JSON object with a
``type`` discriminator.

Fields added later all have defaults and unknown fields are ignored, so agents
and coordinators of different versions still understand each other. ``seq`` and
``Ack`` are only used once both ends agree on a reliable session (see
``common.reliable``)."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

from slashcompute.jobs.lora_finetune import LoraFinetuneSpec


class DeviceProfile(BaseModel):
    chip: str
    memory_total_bytes: int
    memory_available_bytes: int
    working_set_bytes: int
    # What the contributor lends: what is free, or the --max-memory-gb choice (at most the working set).
    memory_contrib_bytes: int
    matmul_tflops: float
    mem_bandwidth_gbps: float


class UsageSample(BaseModel):
    """Raw per-step measurements for one stage. Credits post from these when a job is reserved."""

    flops: float
    tokens: int
    peak_mem_bytes: int
    resident_mem_bytes: int
    mem_byte_seconds: float
    wall_s: float
    busy_s: float


class PeerAddr(BaseModel):
    node_id: str
    host: str
    port: int


class Msg(BaseModel):
    """Base of every control message. ``seq`` numbers it within a reliable session."""

    seq: Optional[int] = None


class Ack(Msg):
    """Both directions: every sequenced message up to and including ``upto`` arrived
    and was handled, so the sender can stop keeping it for replay."""

    type: Literal["ack"] = "ack"
    upto: int


# ---------------------------------------------------------------- agent -> coordinator


class Register(Msg):
    type: Literal["register"] = "register"
    node_id: str
    name: str
    device: DeviceProfile
    data_host: str
    data_port: int
    gpu_percent: int
    session_token: Optional[str] = None
    # Reliable session: random per agent process (None from older agents), and the
    # highest coordinator seq it has handled, so a reconnect can resume.
    session_id: Optional[str] = None
    last_seq: int = 0


class Heartbeat(Msg):
    type: Literal["heartbeat"] = "heartbeat"
    node_id: str
    status: Literal["idle", "loading", "running", "draining"]
    job_id: Optional[str] = None
    epoch: Optional[int] = None
    # While "loading": "fetching" means the model is still downloading, with its progress.
    # Optional and a plain str, so older coordinators ignore them and newer phases parse.
    phase: Optional[str] = None
    fetch_done_bytes: Optional[int] = None
    fetch_total_bytes: Optional[int] = None


class DrainNotice(Msg):
    """Contributor asked to stop; the node leaves after its current step."""

    type: Literal["drain_notice"] = "drain_notice"
    node_id: str


class StageReady(Msg):
    type: Literal["stage_ready"] = "stage_ready"
    job_id: str
    epoch: int
    stage_idx: int


class StepMetrics(Msg):
    type: Literal["step_metrics"] = "step_metrics"
    job_id: str
    epoch: int
    stage_idx: int
    step: int
    loss: Optional[float] = None
    # sha256 of the first microbatch's stage input/output bytes, committed at
    # step time so later verification uploads cannot be swapped.
    in_digest: str
    out_digest: str
    usage: UsageSample


class CheckpointReady(Msg):
    type: Literal["checkpoint_ready"] = "checkpoint_ready"
    job_id: str
    epoch: int
    stage_idx: int
    step: int
    layer_start: int
    layer_end: int
    path: str  # local to the worker; the daemon uploads it


class StageFinished(Msg):
    type: Literal["stage_finished"] = "stage_finished"
    job_id: str
    epoch: int
    stage_idx: int
    reason: Literal["done", "drained", "cancelled", "error"]
    last_step: int
    detail: Optional[str] = None
    fatal: bool = False  # deterministic error (e.g. bad dataset): fail the job, don't retry


class VerifyBundleReady(Msg):
    type: Literal["verify_bundle_ready"] = "verify_bundle_ready"
    verify_id: str
    job_id: str
    stage_idx: int
    step: int
    path: Optional[str] = None  # local to the worker; None if no longer held
    error: Optional[str] = None


class VerifyResult(Msg):
    type: Literal["verify_result"] = "verify_result"
    verify_id: str
    kind: Literal["replay", "canary"]
    stats: dict[str, float] = {}
    output_path: Optional[str] = None  # replay output, local to the verifier
    error: Optional[str] = None


# ---------------------------------------------------------------- coordinator -> agent


class Welcome(Msg):
    type: Literal["welcome"] = "welcome"
    node_id: str
    heartbeat_interval_s: float
    session: bool = False             # reliable session on (seq / Ack / replay)
    resumed: bool = False             # the agent's previous connection was picked up again
    last_seq: int = 0                 # highest agent seq the coordinator has handled
    reconnect_grace_s: float = 0.0    # how long the coordinator holds a dropped agent's place


class StageAssignment(Msg):
    type: Literal["stage_assignment"] = "stage_assignment"
    job_id: str
    epoch: int
    stage_idx: int
    num_stages: int
    layer_start: int
    layer_end: int
    num_layers: int
    spec: LoraFinetuneSpec
    prev_peer: Optional[PeerAddr] = None
    next_peer: Optional[PeerAddr] = None
    resume_step: int = 0  # 0 = fresh start; otherwise a complete checkpoint step
    checkpoint_url: Optional[str] = None
    dataset_url: Optional[str] = None  # only for stage 0
    checkpoint_every: int
    verify_ring_size: int
    peer_timeout_s: float = 600.0  # defaulted so assignments from older coordinators parse


class Drain(Msg):
    """Sent to stage 0: finish the current step, checkpoint, propagate STOP."""

    type: Literal["drain"] = "drain"
    job_id: str
    epoch: int


class CancelStage(Msg):
    type: Literal["cancel_stage"] = "cancel_stage"
    job_id: str
    epoch: int


class VerifyFetch(Msg):
    """Ask a stage to upload the input/output/adapters it used at ``step``."""

    type: Literal["verify_fetch"] = "verify_fetch"
    verify_id: str
    job_id: str
    epoch: int
    stage_idx: int
    step: int


class VerifyRequest(Msg):
    type: Literal["verify_request"] = "verify_request"
    verify_id: str
    kind: Literal["replay", "canary"]
    # replay
    model: Optional[str] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    num_layers: Optional[int] = None
    lora_rank: Optional[int] = None
    lora_scale: Optional[float] = None
    lora_targets: Optional[list[str]] = None
    bundle_url: Optional[str] = None
    # canary
    seed: Optional[int] = None
    size: Optional[int] = None


AgentMessage = Annotated[
    Union[
        Register, Heartbeat, DrainNotice, StageReady, StepMetrics, CheckpointReady,
        StageFinished, VerifyBundleReady, VerifyResult, Ack,
    ],
    Field(discriminator="type"),
]

CoordinatorMessage = Annotated[
    Union[Welcome, StageAssignment, Drain, CancelStage, VerifyFetch, VerifyRequest, Ack],
    Field(discriminator="type"),
]

_agent_adapter = TypeAdapter(AgentMessage)
_coord_adapter = TypeAdapter(CoordinatorMessage)


def parse_agent_message(raw: str | bytes | dict):
    if isinstance(raw, dict):
        return _agent_adapter.validate_python(raw)
    return _agent_adapter.validate_json(raw)


def parse_coordinator_message(raw: str | bytes | dict):
    if isinstance(raw, dict):
        return _coord_adapter.validate_python(raw)
    return _coord_adapter.validate_json(raw)


def dump(msg: BaseModel) -> str:
    return msg.model_dump_json()
