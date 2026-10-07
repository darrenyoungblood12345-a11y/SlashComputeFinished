"""SQLite schema for inference (models, nodes, pipelines, jobs). Credits live in the main coordinator DB."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY,                 -- GGUF file name (first shard)
  sha256 TEXT, size_bytes INTEGER, arch TEXT, moe INTEGER,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending | ready | rejected | disabled (not served) | removed
  status_reason TEXT,
  layer_source TEXT,                   -- node:<id> | synthetic
  layout_json TEXT, flops_json TEXT,
  created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS model_layers (
  model_id TEXT, idx INTEGER, full_bytes INTEGER, active_bytes INTEGER,
  kv_bytes_per_token INTEGER, kv_window INTEGER,
  PRIMARY KEY (model_id, idx)
);
CREATE TABLE IF NOT EXISTS model_files (
  name TEXT PRIMARY KEY,               -- GGUF uploaded to the coordinator, pushed to heads
  size INTEGER, sha256 TEXT, uploaded_at REAL
);
CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY, name TEXT, token_hash TEXT,
  os TEXT, chip TEXT, total_mem_bytes INTEGER,
  committed_bytes INTEGER, hours_json TEXT, allowed_models_json TEXT, can_head INTEGER,
  device TEXT, tailscale_ip TEXT, probe_port INTEGER, llama_build TEXT, gguf_files_json TEXT,
  prompt_score REAL, gen_score REAL,
  reliability REAL NOT NULL DEFAULT 1.0,
  draining INTEGER NOT NULL DEFAULT 0, available INTEGER NOT NULL DEFAULT 1,
  approved_head INTEGER NOT NULL DEFAULT 0, rtt_ms REAL,
  downloads_json TEXT,                 -- {filename: fraction done} while fetching uploaded models
  last_heartbeat REAL, created_at REAL
);
CREATE TABLE IF NOT EXISTS node_benchmarks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT, at REAL,
  prompt_score REAL, gen_score REAL, raw_json TEXT
);
CREATE TABLE IF NOT EXISTS latency (
  a TEXT, b TEXT, one_way_ms REAL, measured_at REAL, PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS pipelines (
  id TEXT PRIMARY KEY, model_id TEXT, ctx INTEGER, state TEXT, head_node_id TEXT,
  plan_json TEXT, explanation TEXT, tensor_split TEXT, est_tok_s REAL, live_tok_s REAL,
  device_list TEXT, load_seconds REAL, broken_reason TEXT,
  created_at REAL, active_at REAL, last_used_at REAL, stopped_at REAL
);
CREATE TABLE IF NOT EXISTS pipeline_members (
  pipeline_id TEXT, node_id TEXT, position INTEGER, role TEXT,
  layer_start INTEGER, layer_end INTEGER, full_bytes INTEGER, active_bytes INTEGER, kv_bytes INTEGER,
  share TEXT, endpoint TEXT,
  PRIMARY KEY (pipeline_id, node_id)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, requester_id TEXT, model_id TEXT, pipeline_id TEXT,
  state TEXT,                               -- queued | running | done | failed | cancelled
  attempt INTEGER NOT NULL DEFAULT 1, retry_of TEXT, stream INTEGER,
  prompt_n INTEGER, cache_n INTEGER, predicted_n INTEGER,
  flops REAL, gen_weight REAL, tok_s REAL, error TEXT, retryable INTEGER,
  request_json TEXT, output_head TEXT,
  created_at REAL, finished_at REAL
);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def tx(conn: sqlite3.Connection):
    """One atomic transaction (connections run in autocommit mode otherwise)."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
