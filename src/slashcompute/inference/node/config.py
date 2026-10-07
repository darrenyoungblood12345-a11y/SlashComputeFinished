"""Inference node configuration (built from CLI flags by ``slashcompute.inference.node``)."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BASKET = "Qwen3.5-4B-Q4_K_M.gguf"


@dataclass(frozen=True)
class Commitment:
    memory_gb: float = 8.0                            # GiB of unified memory lent to inference
    hours: tuple[tuple[int, int], ...] = ((0, 24),)   # allowed UTC hour ranges [start, end)
    allowed_models: tuple[str, ...] = ()              # empty = any model
    may_be_head: bool = False
    device: str = ""                                  # e.g. MTL0 (empty = default device)

    def to_json(self) -> dict:
        return {"memory_gb": self.memory_gb, "hours": [list(h) for h in self.hours],
                "allowed_models": list(self.allowed_models) or None, "may_be_head": self.may_be_head,
                "device": self.device or None}


@dataclass(frozen=True)
class NodeConfig:
    coordinator_url: str = "http://127.0.0.1:8765/inference"
    name: str = "node"
    llama_server: str = "llama-server"
    rpc_server: str = "rpc-server"
    models_dir: str = "~/models"
    download_dir: str = "~/.slashcompute/models"      # uploaded models pushed by the coordinator land here
    basket_model: str = DEFAULT_BASKET
    basket_ref_prompt_tps: float = 2000.0
    basket_ref_gen_tps: float = 110.0
    probe_port: int = 0                               # 0 = any free port (reported at registration)
    state_file: str = "~/.slashcompute/inference/node_state.json"
    status_file: str = "~/.slashcompute/inference/status.json"
    agent_status_file: str = "~/.slashcompute/agent/status.json"   # the MLX agent: training wins the Mac
    ip: str = ""                                      # LAN address for direct transport
    benchmark: str = "auto"                           # auto: run the join benchmark once; skip: assumed scores
    assumed_gen_score: float = 0.3
    assumed_prompt_score: float = 0.3
    session_token: str = ""                           # owner's /compute session: earnings go to them
    inference_token: str = ""                         # shared secret when the coordinator requires one
    # a model removed from the pool: move a copy in models_dir to the Trash (LAN pools), or keep it (public pools)
    trash_removed: bool = False
    commitment: Commitment = field(default_factory=Commitment)

    @property
    def model_dirs(self) -> list[str]:
        return list(dict.fromkeys([self.models_dir, self.download_dir]))
