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
- **Remove** asks first, naming the Macs that hold the model, those that are offline and those
  still downloading it, or saying that no Mac has a copy. Then it:
  - unloads the model (a reply in progress finishes first) and takes it out of the picker;
  - deletes the copy uploaded to the pool, and stops sending it to Macs at once;
  - on each Mac that has it, deletes the copies this pool sent there (into `~/.slashcompute/models`).
    Any other copy goes to that Mac's Trash on a LAN pool, so Finder's **Put Back** restores it, and
    stays where it is on a public pool, since that pool's coordinator belongs to someone else. "Any
    other copy" is one in the Mac's own models folder (`~/models` unless changed), or one in
    `~/.slashcompute/models` that this pool did not send: the Mac's own upload when it hosts a pool,
    another pool's, or one sent before the Mac's app kept a record of what each pool sent. A split
    model goes with all its parts (`-00001-of-00003.gguf` and the rest);
  - keeps the model from coming back by itself: a Mac that still reports the file does not add it
    back.

  A Mac records which copies each pool sent it under that pool's address, in
  `~/.slashcompute/inference/downloaded.json`. If the hosting Mac's address changes, the copies it
  sent before count as not sent by this pool. A copy written over since it was sent (by an upload of
  the same name on this Mac, say) does too.

  Your own uploads sit in the same folder, `~/.slashcompute/models`. So when this Mac joins someone
  else's LAN pool and that pool removes a model with the same file name, your upload goes to the
  Trash. **Put Back** restores it; until then your own pool lists the model but can't serve it.

  The row stays while a copy is left, with a line for each:
  - `Removing from <Mac>…`: the Mac is deleting it now.
  - `<Mac> is offline and keeps its copy. Press Remove again once it is back.` Pressing **Remove
    again** while that Mac is still offline stops waiting for it: the row goes. When the Mac is back
    with its copy, the row comes back, still removed (a Mac on an older /compute shows again only once
    it restarts serving).
  - `An old record of <Mac> still lists a copy. Press Remove again.`: the Mac rejoined the pool under a
    new identity and is online; the pool still has its old record. **Remove again** clears it.
  - `<Mac>: did not answer in time`: the Mac was asked but did not reply within two minutes (it went
    to sleep or lost its connection, say). Press **Remove again** once it is reachable.
  - `<Mac>: node <Mac> went offline`: it went offline while removing. Press **Remove again** once it
    is back.
  - `<Mac>: runs an older /compute: update it, or delete <file> by hand from its models folder or
    from ~/.slashcompute/models`: the pool can't tell which of the two folders holds it.
  - `<Mac>: <file>: <reason>`: the move to the Trash or the delete failed (permissions, for
    example). The file stays where it was; fix the cause and press **Remove again**.
  - `Kept in <Mac>'s own models folder. Delete it there by hand if you want it gone.`: that Mac does
    not let the pool move its files to the Trash. Macs on a public pool never do.
  - `Kept in <Mac>'s app folder (~/.slashcompute/models): this pool has no record of sending that
    copy. Delete it there by hand if you want it gone.`: the same, for a copy in the app's folder that
    this pool has no record of sending (see above).
  - `Still on <Mac>. Press Remove again.`: the Mac has a copy, but the pool has no record of
    removing it there. That is a Mac that was offline and is back, a download that finished after
    Remove, or any Mac after the pool's host restarted (it does not keep how each removal went).
  - `The pool's uploaded copy could not be deleted: <reason>. Press Remove again.`: the hosting Mac
    could not delete its own copy (a locked file or a changed folder permission, say). Fix the cause
    and press **Remove again**. After the host restarts it no longer knows the reason, and the line
    reads `The pool's uploaded copy is still there. Press Remove again.`
- **Add back** (on a removed model whose layers the pool still knows) lists it again for the Macs
  that still have the file. Uploading the GGUF again does the same. A removed model never returns by
  itself: a file put back from the Trash shows again as removed, with **Add back**.

Each Mac looks at its models folders every few seconds, so a GGUF dropped in, moved to the Trash,
deleted or put back shows or goes without restarting anything. A file still being copied in shows
once it is whole. A Mac running an older /compute reads its folders only when its LLM serving
starts: after changing its files by hand, stop and start serving on that Mac.

When a button can't do it, the toast says why:

| The toast says | Why | Fix |
|---|---|---|
| `<model> is still being removed from the pool's Macs; try again in a moment` | An upload of a model whose removal is still running | Wait for the `Removing from` lines to go, then upload again |
| `<model> has no usable layer table: upload it again` | **Serve** or **Add back** on a model the pool never managed to read | Upload the GGUF again |
| `<model> is <status>: there is nothing to stop serving` | **Stop serving** on a model that was removed, rejected or never read | **Remove** it instead |
| `Only an admin can manage models and pipelines.` | A public pool, and this account is not its admin | Ask the pool's admin |
| `Removed <model>. The pool's uploaded copy could not be deleted: <reason>.` | The hosting Mac could not delete its own copy (a locked file, a changed folder permission) | Fix the cause, then **Remove again** on the model's row |

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
