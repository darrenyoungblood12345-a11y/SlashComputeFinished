import pytest
from pydantic import ValidationError

from slashcompute.common import protocol as P
from slashcompute.jobs import LoraFinetuneSpec, parse_spec


def _device():
    return P.DeviceProfile(
        chip="Apple M5", memory_total_bytes=16 << 30, memory_available_bytes=8 << 30,
        working_set_bytes=12 << 30, memory_contrib_bytes=6 << 30,
        matmul_tflops=10.0, mem_bandwidth_gbps=120.0,
    )


def test_agent_message_roundtrip():
    msg = P.Register(node_id="n1", name="mac", device=_device(), data_host="10.0.0.2",
                     data_port=9700, gpu_percent=50)
    parsed = P.parse_agent_message(P.dump(msg))
    assert isinstance(parsed, P.Register)
    assert parsed == msg


def test_coordinator_message_roundtrip():
    spec = LoraFinetuneSpec(dataset_path="/tmp/d.jsonl", steps=3)
    msg = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=2, layer_start=0, layer_end=4,
        num_layers=8, spec=spec, next_peer=P.PeerAddr(node_id="b", host="h", port=1),
        checkpoint_every=25, verify_ring_size=8,
    )
    parsed = P.parse_coordinator_message(P.dump(msg))
    assert isinstance(parsed, P.StageAssignment)
    assert parsed.spec.steps == 3 and parsed.next_peer.port == 1


def test_unknown_message_rejected():
    with pytest.raises(ValidationError):
        P.parse_agent_message({"type": "nope"})


def test_spec_validation():
    with pytest.raises(ValidationError):
        LoraFinetuneSpec(dataset_path="x", batch_size=3, microbatches=2)
    with pytest.raises(ValueError):
        parse_spec({"kind": "run_arbitrary_code", "dataset_path": "x"})
    with pytest.raises(ValueError, match="not allowed"):
        parse_spec({"dataset_path": "x", "model": "evil/malware-repo"})
    assert parse_spec({"dataset_path": "x"}).kind == "lora_finetune"


def test_heartbeat_progress_fields_are_optional():
    """Older agents send none of them; older coordinators ignore them."""
    old = P.parse_agent_message({"type": "heartbeat", "node_id": "n", "status": "loading"})
    assert old.phase is None and old.fetch_done_bytes is None and old.fetch_total_bytes is None
    new = P.parse_agent_message({"type": "heartbeat", "node_id": "n", "status": "loading",
                                 "job_id": "j", "epoch": 1, "phase": "fetching",
                                 "fetch_done_bytes": 5, "fetch_total_bytes": 9, "later_field": 1})
    assert (new.phase, new.fetch_done_bytes, new.fetch_total_bytes) == ("fetching", 5, 9)
