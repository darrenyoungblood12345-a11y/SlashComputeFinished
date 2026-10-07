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

The first message to a model loads it. **Send** works once a Mac in the pool is serving the
selected model. When it can't, the box under it says why, and usually has the button that fixes it:

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
| `<model>` is not being served: it was stopped for the whole pool | Someone pressed **Stop serving** on it | **Serve**, in the box or under Models. On a public pool only its admin can |

If the node exits on start, the toast says why, e.g.
`The LLM node exited with code 1: llama.cpp RPC server not found ...`. The full output is in
`inference.log`.

A chat sent anyway, from the API or from a window opened before the change, is refused:

| The error says | Why | Fix |
|---|---|---|
| `<model> is not being served: it was stopped in the LLMs tab. Press Serve there to use it again.` | The model's Serve switch is off (HTTP 404) | **Serve** |
| `<model> was removed from this pool.` | The model was removed (HTTP 404) | **Add back**, or upload it again |

### Unload, Stop serving and Remove

Each row under **Models** has the model's buttons. Anyone can use them on a LAN pool; on a public
pool only an admin can, and everyone else sees a note saying so. A pool whose coordinator runs an
older app shows a note instead: update the hosting Mac.

- **Unload** (while the model is loaded or loading) stops every pipeline of that model across the
  pool. A reply in progress finishes first, and the row says `unloading` meanwhile. It is not
  sticky: the next chat loads the model again, and a chat already waiting for a model that was
  loading loads it again straight away. The Pipeline card's **Unload** does the same.
- **Stop serving** is a switch for the whole pool. It unloads the model on every Mac and keeps it
  from loading until **Serve** is pressed. Its files stay on disk. The switch is kept by the
  coordinator, so it survives restarts and Macs joining again. The row says `not served`, the
  model picker says `· stopped`, and the box under Send offers **Serve**.
- **Remove** asks first, naming the Macs that hold the model and those that are offline, or
  saying that no Mac has a copy. Then it:
  - unloads the model (a reply in progress finishes first) and takes it out of the picker;
  - deletes the copy uploaded to the pool and the app's own copies on the Macs
    (`~/.slashcompute/models`);
  - moves a copy in a Mac's own models folder (`~/models` unless changed) to that Mac's Trash, so
    Finder's **Put Back** restores it. A split model goes with all its parts
    (`-00001-of-00003.gguf` and the rest);
  - keeps the model from coming back by itself: a Mac that still reports the file does not add it
    back.

  The row stays while a Mac still holds a copy, with a line per Mac:
  - `Removing from <Mac>…`: the Mac is deleting it now.
  - `<Mac> is offline and keeps its copy`: press **Remove again** once that Mac is back.
  - `<Mac>: runs an older /compute: update it, or delete <file> from its models folder by hand`.
  - `<Mac>: <file>: <reason>`: the move to the Trash failed (permissions, for example). The file
    stays where it was; fix the cause and press **Remove again**.
  - `Kept in <Mac>'s own models folder`: that Mac does not let the pool move its files to the
    Trash. Macs on a public pool never do, since that pool's coordinator belongs to someone else.
    Delete the file there by hand.
- **Add back** (on a removed model whose layers the pool still knows) lists it again for the Macs
  that still have the file. Uploading the GGUF again does the same. A file put back from the Trash
  by hand also needs **Add back**: a removed model never returns by itself.

When a button can't do it, the toast says why:

| The toast says | Why | Fix |
|---|---|---|
| `<model> is still being removed from the pool's Macs; try again in a moment` | An upload of a model whose removal is still running | Wait for the `Removing from` lines to go, then upload again |
| `<model> has no usable layer table: upload it again` | **Serve** or **Add back** on a model the pool never managed to read | Upload the GGUF again |
| `<model> is <status>: there is nothing to stop serving` | **Stop serving** on a model that was removed, rejected or never read | **Remove** it instead |
| `Only an admin can manage models and pipelines.` | A public pool, and this account is not its admin | Ask the pool's admin |

None of these clears llama.cpp's RPC tensor cache, `~/Library/Caches/llama.cpp/rpc` on each Mac
that lent layers. It can hold several GB. Delete the folder by hand to get the space back;
llama.cpp fills it again on the next load.

**Stop serving** on the This Mac card is different: it stops the LLM node on this Mac only, and
straight away. Replies this Mac was taking part in fail.

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
