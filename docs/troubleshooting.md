# Troubleshooting

What the app shows, why, where to look, and what to do. Logs are in `~/.slashcompute/logs/`:
`coordinator.log` (the pool, on the hosting Mac), `agent.log` (this Mac's training agent) and
`inference.log` (this Mac's LLM node).

## Training jobs

### A job says "starting" for a long time

A Mac given a job first downloads the model from Hugging Face. A 7B model is about 4 GB, a 14B
about 8 GB, and every Mac in the job downloads the whole model once. Later jobs with the same
model start in seconds.

- **What you see:** the job card says `Downloading the model on <Mac>: 1.2 GB of 4.0 GB (30%)`,
  and the Contribute tab says `Agent is downloading the model`.
- **Log:** `agent.log` has `job <id>: fetching <model>`, then `fetched X of Y GB` every 30 s,
  then `<model> is ready`.
- **What to do:** wait, or cancel. Cancelling stops the download within a couple of seconds and
  frees the Mac. The part already downloaded is thrown away.

A start is only given up when it stops making progress for `SLASHCOMPUTE_STAGE_START_TIMEOUT_S`
(900 s): no new bytes and no stage becoming ready. The card then says
`Last try: stages not ready: no progress for 900s`, and the job is retried. It prefers other
Macs for `SLASHCOMPUTE_START_TIMEOUT_AVOID_S` (600 s), but uses the same one if it is the only one.

### A download fails

- **What you see:** `Last try: stage 0 error: model fetch failed: ...` on the job card.
- **Log:** `agent.log` has `model fetch for job <id> failed: <reason>`.
- **Common reasons:**
  - A `CAS Client Error` or a timeout: the network or Hugging Face had trouble. The job retries.
  - Rate limits: set `HF_TOKEN` in the environment of the app that runs the agent.
  - Not enough disk space: Hugging Face checks before it downloads and says so.

### A job waits with "needs 2 Macs; the pool has 1"

- **Why:** the job's **Min Macs** is higher than the number of Macs in the pool. It waits until
  enough Macs join, and it no longer blocks the jobs behind it.
- **What to do:** cancel it and submit again with Min Macs 1 (the default), or add a Mac.
  A model too big for one Mac is still split across Macs whatever Min Macs says.

### A job waits with "waiting for <Mac> to stop a cancelled job"

- **Why:** a cancelled job's Mac is free again only once its agent confirms it stopped. Before
  this rule, a new job went to a Mac still downloading the cancelled job's model and queued behind
  it, so it said "starting" and never ran.
- **Log:** `coordinator.log` has `job <id> cancelled (...)`, then `node <id> let go of job <id>`,
  usually within a second or two.
- **If it lasts:** an agent from an older app may never confirm. The Mac is freed after
  `SLASHCOMPUTE_RELEASE_TIMEOUT_S` (60 s), unless its heartbeats still say it is busy with the
  cancelled job (`is still busy with cancelled job ... holding it`). Update the app on that Mac,
  or stop and start its agent.

### A job is refused with "dataset line N ..."

The dataset is checked when you submit it. It needs one JSON object per line, with `text`,
`prompt` + `completion`, `messages` or `tokens`. The message names the first bad line.

### "Training agent is still stopping"

The agent finishes or checkpoints its current step before it stops. Changing a training setting
(GPU share, memory, pool address) restarts it. Starting or stopping LLM serving does not touch it.

## LLMs tab

The first message to a model loads it, so there is no separate "load" button. **Send** works once
a Mac in the pool is serving the selected model. When it can't, the box under it says why, and
usually has the button that fixes it:

| The box says | Why | Fix |
|---|---|---|
| Start or join a pool first | No coordinator | Pool tab |
| No models yet | Nothing uploaded and no head has a GGUF in its models folder | Upload a GGUF |
| No Mac is serving `<model>` yet | No Mac runs the LLM node | **Start serving on this Mac** |
| Starting llama.cpp on this Mac… | The node is joining the pool | Wait a few seconds |
| This Mac only lends layers | Role is "Worker only" | **Make this Mac a head** |
| Copying `<model>` to this Mac | An upload is being pushed to this head | Wait |
| `<model>` needs N GB lent to run on one Mac | Not enough memory lent | **Lend N GB and restart serving**, or add a Mac |
| Not taking work right now: training on this Mac | A training job runs here | Wait for it, or serve from another Mac |
| Can't reach the pool's LLM service | The coordinator answered with an error | The message has the details |
| The pool's coordinator has no LLM inference | The pool runs an older app | Update the hosting Mac |

If the node exits on start, the toast says why, e.g.
`The LLM node exited with code 1: llama.cpp RPC server not found ...`. The full output is in
`inference.log`.

### Joining a pool by URL

The address you joined last is kept in `~/.slashcompute/launcher.json` (`"url"`). Hosting ignores
it. An LLM node pointed at an old pool logs `coordinator not reachable yet (... 404 ...
/inference/ping)` forever: that pool's app predates LLM inference.

## Is the app running the code you think?

The app carries its own copy of `/compute`. A fix in the repository reaches it only when the app
is rebuilt (`scripts/macos/build_dmg.sh`) and reinstalled. To check:

```bash
diff -rq --exclude=__pycache__ /Applications/compute.app/Contents/Resources/python/lib/python3.13/site-packages/slashcompute src/slashcompute
```

The window asks for `app.js` by content version and revalidates it, so an update shows on the
next load. A running shell from an older version is replaced, not reused.
