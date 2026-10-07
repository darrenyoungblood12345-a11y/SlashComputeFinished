# /compute

Pooling engine: pipeline-parallel LoRA fine-tunes across Apple Silicon Macs on a LAN. Contributors run a headless agent; one coordinator schedules stages, records usage, and checks work.

Accounts, credits, and grants are later. This build is the Mac engine plus a local app.

## Install

[Full .dmg File is Here.](https://drive.google.com/open?id=1lpUIbXkPRyN-sXPkNIsoMFWbBo1BreIu)

## Public-pool access

With `SLASHCOMPUTE_PUBLIC_POOL=1`, only a job's owner or an administrator can cancel it or download its output adapter. Dataset and checkpoint downloads also allow authenticated contributors currently assigned to that job. Checkpoint uploads and verification transfers require the matching assignment. Agents use `--session-token` (or `SLASHCOMPUTE_SESSION`) for both registration and HTTP transfers, including sandboxed workers.

Public users submit training datasets through `POST /jobs/upload` (the app's Usage form). Submitting coordinator-local dataset paths through `POST /jobs` is restricted to administrators. Temporary uploads are removed after submission, and rejected reservations discard their dataset copies.

Each account gets a one-time welcome credit of 1 PFLOP (1e15 FLOPs) when it first signs in (email or Google); set `SLASHCOMPUTE_WELCOME_FLOPS` on the coordinator to change it, or `0` to disable.

Changing the pool address, GPU share, or session restarts the training agent after it drains. If it is still stopping, the app reports that settings have not yet been applied; start again after the current work finishes. Existing agents without saved argument metadata rejoin once after upgrading.


## How a job runs

1. Agents register, pass a short GPU canary, and heartbeat.
2. The coordinator splits the model by layer, sized to each Mac's contributed memory.
3. Each Mac downloads the model the first time. The job card shows the progress, and cancelling stops the download at once. A Mac whose job was cancelled gets new work only after its agent confirms it stopped.
4. Neighbouring stages open a TCP link and run a GPipe LoRA step: activations forward, gradients back.
5. Each step is metered (FLOPs, memory, time).
6. `stop` drains after the current step; a crash resumes from the last complete checkpoint.

Something stuck or unclear? See [docs/troubleshooting.md](docs/troubleshooting.md).

## When the network drops

- **Between stages:** peer links use TCP keepalive and time out instead of waiting forever. A dropped link redials and resends the activations or gradients the other stage missed, so a short outage doesn't restart the job. A neighbour that stays silent for `SLASHCOMPUTE_PEER_TIMEOUT_S` (600 s) fails the stage, and the job resumes from its last checkpoint.
- **Between an agent and the coordinator:** both sides number their control messages and resend whatever wasn't acknowledged. An agent that drops off (or stops heartbeating) and returns within `SLASHCOMPUTE_RECONNECT_GRACE_S` (60 s) keeps its stage. Agents and coordinators from before this change keep the old behaviour with each other.
- **Backstop:** a running job that reports no step for `SLASHCOMPUTE_STALL_TIMEOUT_S` (1800 s) restarts from its last checkpoint.
