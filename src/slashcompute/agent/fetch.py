"""Download a stage's model into the Hugging Face cache, in a process the agent can kill.

The agent used to download in a thread its message loop waited on: a cancel, a newer
assignment or a shutdown sat unread for the whole download (10-40 min for a 7-14B
model), and a thread can't be stopped. Run as ``python -m slashcompute.agent.fetch
<model>``, so stopping a download is a SIGTERM. It writes JSON lines on stdout:

    {"type": "progress", "done": <bytes>, "total": <bytes>}
    {"type": "done", "path": "<snapshot dir>"}
    {"type": "error", "detail": "<why>"}
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable

from slashcompute.pipeline.model_profile import MODEL_FILE_PATTERNS, resolve_model_path

PROGRESS_EVERY_S = 1.0


def _cache_dirs(model: str) -> tuple[Path, Path]:
    """The repo's cache folder and its lock folder."""
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    name = repo_folder_name(repo_id=model, repo_type="model")
    return Path(constants.HF_HUB_CACHE) / name, Path(constants.HF_HUB_CACHE) / ".locks" / name


def sweep_orphans(model: str) -> list[Path]:
    """Delete partial downloads nobody is writing any more.

    A killed download leaves ``blobs/<etag>.<id>.incomplete`` behind, and Hugging Face never
    resumes another process's file. It writes a blob while holding the flock
    ``.locks/<repo>/<etag>.lock``, which dies with its process: a lock we can take means the
    partial file is an orphan."""
    from filelock import FileLock, Timeout

    repo, locks = _cache_dirs(model)
    removed = []
    for part in sorted((repo / "blobs").glob("*.incomplete")):
        lock = locks / f"{part.name.split('.', 1)[0]}.lock"
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(lock), timeout=0):
                part.unlink(missing_ok=True)
            removed.append(part)
        except (Timeout, OSError):
            continue                      # another download is writing it
    return removed


def expected_blobs(model: str) -> dict[str, int]:
    """Size of each file the download fetches, by the name the cache stores it under
    (``blobs/<etag>``: the LFS sha256, else the git blob id)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(model, files_metadata=True)
    out = {}
    for s in filter_repo_objects(info.siblings or [], allow_patterns=MODEL_FILE_PATTERNS,
                                 key=lambda s: s.rfilename):
        etag = s.lfs.sha256 if s.lfs else s.blob_id
        size = s.lfs.size if s.lfs else s.size
        if etag and size is not None:
            out[etag] = int(size)
    return out


def downloaded_bytes(blobs: Path, expected: dict[str, int], ignore: frozenset[str] = frozenset()) -> int:
    """Bytes of ``expected`` on disk: finished blobs plus partial ones not in ``ignore``."""
    done = sum(size for etag, size in expected.items() if (blobs / etag).exists())
    for part in blobs.glob("*.incomplete"):
        etag = part.name.split(".", 1)[0]
        if part.name in ignore or etag not in expected or (blobs / etag).exists():
            continue
        try:
            st = part.stat()
        except OSError:
            continue                      # finished (renamed into place) meanwhile
        # Allocated bytes, not the length: a download may size its file up front.
        done += min(st.st_size, st.st_blocks * 512)
    return min(done, sum(expected.values()))


def _report(model: str, emit: Callable[..., None], stop: threading.Event) -> None:
    try:
        expected = expected_blobs(model)
    except Exception:
        return                            # offline or private: no byte counts, still "fetching"
    blobs = _cache_dirs(model)[0] / "blobs"
    # Partial files another process is writing (a sweep left them) are not ours.
    ignore = frozenset(p.name for p in blobs.glob("*.incomplete"))
    total, last = sum(expected.values()), None
    while True:
        done = downloaded_bytes(blobs, expected, ignore)
        if done != last:
            emit(type="progress", done=done, total=total)
            last = done
        if stop.wait(PROGRESS_EVERY_S):
            return


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m slashcompute.agent.fetch <model>", file=sys.stderr)
        return 2
    model = argv[0]
    # The protocol keeps the real stdout; anything a library prints goes to the agent's log.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    # Stop at once: a normal exit would first wait for the download's threads.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: os._exit(143))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    lock = threading.Lock()

    def emit(**msg) -> None:
        with lock:
            out.write(json.dumps(msg) + "\n")

    try:
        sweep_orphans(model)
    except Exception as e:                # never a reason not to download
        print(f"could not sweep partial downloads: {e}", file=sys.stderr)
    stop = threading.Event()
    threading.Thread(target=_report, args=(model, emit, stop), daemon=True).start()
    try:
        path = resolve_model_path(model)
    except Exception as e:
        emit(type="error", detail=f"{type(e).__name__}: {e}")
        return 1
    finally:
        stop.set()
    emit(type="done", path=str(path))
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)                        # don't wait on the reporter or download threads
