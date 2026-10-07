# Lessons

- Fetch remote refs and compare app branches before reviewing the latest GitHub code; the repository's default branch can lag main. State the exact reviewed commit.
- When restricting private training endpoints, preserve authenticated assigned-agent data transfers, including sandboxed subprocesses.
- Test helpers must isolate every default path (status, state, downloads, models), not only when a test opts in;
  check ~/.slashcompute for files changed by a test run.
- Cached renders (renderOnce) must key on everything their markup reads, including empty-state text.
- On macOS, Python 3.13 skips .pth files with the UF_HIDDEN flag; a venv under a dot-directory (.claude/worktrees) can get it, so the editable install silently vanishes for subprocesses ("No module named slashcompute"). Fix: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.
- Parallel fixer agents: worktrees start from the default branch, not the integration branch; tell them the exact tip to reset to.
- Never resolve source conflicts by keeping both sides; a "union" is only safe when each side adds whole new top-level functions. Merge tests by function blocks and check imports.
- Many agents running test suites at once overload the machine (load ~127 on 10 cores) and make timing tests flaky; have fixers run only related tests and run the full suite once at the end.
- Stress concurrency code in a loop (hundreds of runs) before trusting it; a single passing run of the resilient link hid a 1-in-25 hang.
- Don't rely on the peer's RST: macOS can ignore it under zero-window and only notice at the next persist probe (~5 s). Bound retransmission (TCP_RXT_CONNDROPTIME / TCP_USER_TIMEOUT) and abort a link once its read side ends.
- `mx.save_safetensors` appends `.safetensors` to a path that lacks it; temp files for atomic writes must keep the extension.
- Network drops are usually silent: a reconnect grace window must also cover missed heartbeats, not just socket closes.
- Never await long I/O (a model download) inside a message loop: everything queued behind it, cancels
  included, goes unread. Run it as a task, and as a child process when it must be stoppable.
- Free a remote resource only once the remote side confirms it let go (with a bounded fallback); freeing
  it on our own decision sent new jobs to a Mac still busy with the cancelled one.
- A fix counts once it is proven in the app the user runs: rebuild and reinstall it, diff the installed
  package against src, and make sure the window can't keep serving a cached app.js.
- Start from the user's own logs and databases (`~/.slashcompute/logs`, `sqlite3 -readonly
  coordinator.db`): they showed the real cause where earlier fixes guessed.
