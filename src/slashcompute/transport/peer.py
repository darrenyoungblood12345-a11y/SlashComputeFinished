"""Direct TCP links between neighbouring pipeline stages.

Each link is one TCP connection. A background task reads frames into a queue
so a stage can keep receiving while its own compute or sends are in flight.
The downstream stage listens on its data port and the upstream stage dials
it. A ``hello`` frame binds the connection to a (job, epoch) so stale stages
can't cross-talk.

Real LANs drop packets, sleep laptops and lose Wi-Fi, so nothing here waits
forever: sockets use TCP keepalive (a vanished peer is noticed in about
``KEEPALIVE_IDLE_S + KEEPALIVE_INTERVAL_S * KEEPALIVE_COUNT`` seconds even when
no data is moving) and stop retransmitting undeliverable data after
``RETRANSMIT_DROP_S``, and receives, sends and dials take timeouts. When both
ends support it, a link is a ``ResilientLink``: a dropped connection is redialled
and the frames the peer missed are retransmitted, so a network blip doesn't cost
the job an epoch.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
from collections import OrderedDict
from typing import Awaitable, Callable, Optional

from slashcompute.transport.serialization import (
    Frame, _HDR, decode_header, encode, tensor_from_bytes,
)

log = logging.getLogger(__name__)

_CLOSED = object()

# The peer's kernel answers keepalive probes, so a stage busy in MLX compute
# never trips them; only a peer that is gone (or a path that is cut) does.
KEEPALIVE_IDLE_S = 15
KEEPALIVE_INTERVAL_S = 5
KEEPALIVE_COUNT = 4
# Keepalive doesn't run while data is unacknowledged, so a sender on a dead path would
# otherwise retransmit for many minutes before the kernel gives up. Drop it sooner; a
# ResilientLink then reconnects.
RETRANSMIT_DROP_S = 60
_TCP_USER_TIMEOUT = getattr(socket, "TCP_USER_TIMEOUT", None)  # Linux, milliseconds
# macOS: <netinet/tcp.h> TCP_RXT_CONNDROPTIME, seconds (not exported by the socket module).
_TCP_RXT_CONNDROPTIME = 0x80 if sys.platform == "darwin" else None
# One dial attempt; a silently dropped SYN would otherwise block for the OS
# connect timeout (about 75 s on macOS) before the overall deadline is checked.
CONNECT_ATTEMPT_S = 10.0
CLOSE_TIMEOUT_S = 5.0


class LinkClosed(ConnectionError):
    pass


class LinkTimeout(LinkClosed):
    """The peer sent nothing, or took nothing, for longer than the link allows."""


def _tune_socket(sock) -> None:
    """Keepalive, and a bound on how long undeliverable data is retransmitted."""
    if sock is None:
        return
    opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    # TCP_KEEPIDLE on Linux; macOS calls the same option TCP_KEEPALIVE.
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    for opt, val in ((idle, KEEPALIVE_IDLE_S),
                     (getattr(socket, "TCP_KEEPINTVL", None), KEEPALIVE_INTERVAL_S),
                     (getattr(socket, "TCP_KEEPCNT", None), KEEPALIVE_COUNT),
                     (_TCP_USER_TIMEOUT, RETRANSMIT_DROP_S * 1000),
                     (_TCP_RXT_CONNDROPTIME, RETRANSMIT_DROP_S)):
        if opt is not None:
            opts.append((socket.IPPROTO_TCP, opt, val))
    for level, opt, val in opts:
        try:
            sock.setsockopt(level, opt, val)
        except OSError as e:
            log.debug("could not set socket option %s: %s", opt, e)


class Link:
    """Bidirectional frame channel. Subclasses provide the byte transport."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self.bytes_sent = 0
        self.bytes_received = 0

    async def send(self, frame: Frame) -> None:
        raise NotImplementedError

    async def recv(self, timeout: Optional[float] = None) -> Frame:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            raise LinkTimeout(f"nothing from the peer in {timeout:g}s") from None
        if item is _CLOSED:
            self._queue.put_nowait(_CLOSED)
            raise LinkClosed("peer link closed")
        if isinstance(item, BaseException):
            self._queue.put_nowait(item)
            raise LinkClosed(f"peer link failed: {item!r}") from item
        return item

    async def close(self) -> None:
        raise NotImplementedError


class TcpLink(Link):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 send_timeout: Optional[float] = None) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer
        self.send_timeout = send_timeout
        self._send_lock = asyncio.Lock()
        _tune_socket(writer.get_extra_info("socket"))
        self._task = asyncio.create_task(self._read_loop())

    @property
    def peername(self):
        return self._writer.get_extra_info("peername")

    async def _read_frame(self) -> Frame:
        (hlen,) = _HDR.unpack(await self._reader.readexactly(_HDR.size))
        header = decode_header(await self._reader.readexactly(hlen))
        tensors = {}
        for spec in header["tensors"]:
            buf = await self._reader.readexactly(spec["nbytes"])
            tensors[spec["name"]] = tensor_from_bytes(spec, buf)
            self.bytes_received += spec["nbytes"]
        self.bytes_received += _HDR.size + hlen
        return Frame(header["kind"], header["meta"], tensors)

    async def _read_loop(self) -> None:
        try:
            while True:
                self._queue.put_nowait(await self._read_frame())
        except (asyncio.IncompleteReadError, OSError):  # EOF, reset, keepalive ETIMEDOUT
            self._queue.put_nowait(_CLOSED)
            # Links never half-close: once the peer is gone, fail our pending sends now
            # rather than leave a drain() waiting on a socket nobody will read.
            self._writer.transport.abort()
        except asyncio.CancelledError:
            self._queue.put_nowait(_CLOSED)
        except Exception as e:  # malformed frame
            log.exception("peer link read failed")
            self._queue.put_nowait(e)

    async def send(self, frame: Frame) -> None:
        chunks = encode(frame)
        async with self._send_lock:
            if self._writer.is_closing():
                raise LinkClosed("peer link closed")
            try:
                for c in chunks:
                    self._writer.write(c)
                    self.bytes_sent += len(c)
                await asyncio.wait_for(self._writer.drain(), self.send_timeout)
            except TimeoutError:
                # Part of a frame may be in flight, so this stream can't carry another;
                # abort rather than close, which would wait for the stuck peer to read.
                self._writer.transport.abort()
                raise LinkTimeout(f"the peer took no data for {self.send_timeout:g}s") from None
            except OSError as e:
                raise LinkClosed(str(e)) from e

    def send_nowait(self, frame: Frame) -> None:
        """Queue a small control frame without waiting for the socket to drain, so a
        read loop can answer without blocking on its own sends. It can't split another
        frame: ``send`` writes all of a frame's chunks without yielding in between."""
        if self._writer.is_closing():
            return
        for c in encode(frame):
            self._writer.write(c)
            self.bytes_sent += len(c)

    def abort(self) -> None:
        """Drop the connection now (no flush); the read loop then reports it closed."""
        self._writer.transport.abort()

    async def close(self) -> None:
        self._task.cancel()
        self._writer.close()
        try:
            # A graceful close flushes what is buffered first; a peer that stopped
            # reading would make that wait forever.
            await asyncio.wait_for(self._writer.wait_closed(), CLOSE_TIMEOUT_S)
        except TimeoutError:
            self._writer.transport.abort()
        except Exception:
            pass


class ResilientLink(Link):
    """A peer link that survives its TCP connection dropping.

    Each data frame carries a sequence number (``meta["_seq"]``) and the sender
    keeps it until the peer acknowledges it with an ``_ack`` frame. When the
    connection breaks, the dialing side redials with a ``resume`` hello naming the
    last frame it received, and the listening side answers with its own. Each side
    then retransmits what the other hasn't received, in order and before anything
    new, so the stage above sees every frame exactly once, in order. If no new
    connection arrives within ``window`` seconds the link closes for good. ``_fin``
    marks a deliberate close, so a stage that finished isn't redialled.
    """

    def __init__(self, tcp: TcpLink, *, window: float,
                 redial: Optional[Callable[[int, float], Awaitable[tuple[TcpLink, dict]]]] = None) -> None:
        super().__init__()
        self._tcp = tcp
        self._ready = tcp            # the connection that has had the backlog retransmitted
        self._window = window
        self._redial = redial        # only the dialing side redials
        self._send_seq = 0
        self._recv_seq = 0
        self._unacked: OrderedDict[int, Frame] = OrderedDict()
        self._send_lock = asyncio.Lock()
        self._replaced = asyncio.Event()
        self._closed = False                      # closed by us
        self._gone = False                        # the peer closed it, or it could not reconnect
        self._down_since: Optional[float] = None  # first failure not yet fully resumed
        self.reconnects = 0
        self._pump_task = asyncio.create_task(self._pump())

    @property
    def peername(self):
        return self._tcp.peername

    @property
    def recv_seq(self) -> int:
        return self._recv_seq

    @property
    def closed(self) -> bool:
        return self._closed or self._gone

    def _give_up(self) -> None:
        self._gone = True
        self._unacked.clear()
        self._queue.put_nowait(_CLOSED)

    async def send(self, frame: Frame) -> None:
        if self.closed:  # don't let a stage think it delivered "done" on a dead link
            raise LinkClosed("peer link closed")
        self._send_seq += 1
        f = Frame(frame.kind, {**frame.meta, "_seq": self._send_seq}, frame.tensors)
        self._unacked[self._send_seq] = f
        async with self._send_lock:
            tcp = self._tcp
            if tcp is not self._ready:
                return  # a resume is retransmitting the backlog, which now includes this frame
            try:
                await tcp.send(f)
            except LinkClosed as e:
                # Kept in _unacked: retransmitted after the reconnect, or recv() reports
                # the link closed once the window runs out.
                log.info("peer link send failed (%s); will retransmit after reconnecting", e)
                tcp.abort()

    async def _pump(self) -> None:
        try:
            while True:
                tcp = self._tcp
                try:
                    f = await tcp.recv()
                except LinkClosed as e:
                    if tcp is not self._tcp:
                        continue  # already replaced by a resumed connection
                    if self._closed or not await self._reconnect(tcp, e):
                        self._give_up()
                        return
                    continue
                if f.kind == "_ack":
                    self._on_ack(f.meta.get("seq", 0))
                    continue
                if f.kind == "_fin":  # the peer closed on purpose: nothing to resume
                    self._give_up()
                    return
                seq = f.meta.pop("_seq", None)
                if seq is None or seq <= self._recv_seq:  # retransmitted after a resume
                    tcp.send_nowait(Frame("_ack", {"seq": self._recv_seq}))
                    continue
                if seq != self._recv_seq + 1:
                    log.warning("peer link skipped from frame %d to %d; reconnecting to resend",
                                self._recv_seq, seq)
                    tcp.abort()
                    continue
                self._recv_seq = seq
                self._queue.put_nowait(f)
                tcp.send_nowait(Frame("_ack", {"seq": seq}))
        except asyncio.CancelledError:
            pass

    def _on_ack(self, seq: int) -> None:
        while self._unacked and next(iter(self._unacked)) <= seq:
            self._unacked.popitem(last=False)

    async def _reconnect(self, dead: TcpLink, why: Exception) -> bool:
        self.reconnects += 1
        log.warning("peer link to %s dropped (%s); %s for up to %.0fs", dead.peername, why,
                    "redialling" if self._redial else "waiting for the peer to redial", self._window)
        dead.abort()
        loop = asyncio.get_running_loop()
        if self._down_since is None:
            self._down_since = loop.time()
        deadline = self._down_since + self._window  # repeated drops share one window
        if self._redial is not None:
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    raise TimeoutError(f"no working connection for {self._window:.0f}s")
                tcp, ack = await self._redial(self._recv_seq, remaining)
            except (OSError, TimeoutError) as e:  # LinkClosed is an OSError
                log.warning("could not reconnect the peer link: %s", e)
                return False
            await self._resume(tcp, int(ack.get("recv_seq", 0)))
        while self._tcp is dead:
            self._replaced.clear()
            remaining = deadline - loop.time()
            if remaining <= 0:
                log.warning("peer did not reconnect within %.0fs", self._window)
                return False
            try:
                await asyncio.wait_for(self._replaced.wait(), remaining)
            except TimeoutError:
                pass
        log.info("peer link resumed (reconnect %d)", self.reconnects)
        return True

    async def attach(self, tcp: TcpLink, peer_recv_seq: int) -> None:
        """Listening side: the upstream redialled with a resume hello."""
        await self._resume(tcp, peer_recv_seq)

    async def _resume(self, tcp: TcpLink, peer_recv_seq: int) -> None:
        if self.closed:
            # A resume that raced our close (or came too late): nobody will read or close it.
            tcp.abort()
            return
        old, self._tcp = self._tcp, tcp
        if old is not tcp:
            old.abort()  # a send stuck on the dead connection fails now instead of at its timeout
        self._replaced.set()
        async with self._send_lock:
            if self._tcp is not tcp:
                return  # replaced again meanwhile
            self._on_ack(peer_recv_seq)
            try:
                for f in list(self._unacked.values()):
                    await tcp.send(f)
            except LinkClosed as e:
                log.info("peer link dropped while retransmitting (%s)", e)
                tcp.abort()
                return
            self._ready = tcp
            self._down_since = None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tcp.send_nowait(Frame("_fin", {}))
        self._pump_task.cancel()
        await self._tcp.close()


class MemoryLink(Link):
    """In-process link, used by the single-process pipeline and tests."""

    def __init__(self) -> None:
        super().__init__()
        self.other: Optional[MemoryLink] = None

    @classmethod
    def pair(cls) -> tuple["MemoryLink", "MemoryLink"]:
        a, b = cls(), cls()
        a.other, b.other = b, a
        return a, b

    async def send(self, frame: Frame) -> None:
        if self.other is None:
            raise LinkClosed("peer link closed")
        self.other._queue.put_nowait(frame)

    async def close(self) -> None:
        if self.other is not None:
            self.other._queue.put_nowait(_CLOSED)
            self.other.other = None
        self.other = None


async def connect(host: str, port: int, hello: dict, timeout: float = 60.0,
                  retry_interval: float = 0.25, send_timeout: Optional[float] = None,
                  resume_window: Optional[float] = None) -> Link:
    """Dial a downstream stage, retrying until it accepts this hello.

    A reject is retried like a refused connection: the port may still be held
    by the downstream's previous-epoch listener, which goes away once that
    stage is cancelled and the new epoch's listener binds.

    With ``resume_window`` the link is offered as resilient; a peer that agrees
    gets a ``ResilientLink`` that redials here after a drop. Older peers don't
    answer the offer and get a plain ``TcpLink``.
    """
    if resume_window is None:
        link, _ = await _dial(host, port, hello, timeout, retry_interval, send_timeout)
        return link
    hello = {**hello, "resilient": 1}
    tcp, ack = await _dial(host, port, hello, timeout, retry_interval, send_timeout)
    if not ack.get("resilient"):
        return tcp

    async def redial(recv_seq: int, window: float) -> tuple[TcpLink, dict]:
        return await _dial(host, port, {**hello, "resume": recv_seq}, window, retry_interval,
                           send_timeout)

    return ResilientLink(tcp, window=resume_window, redial=redial)


async def _dial(host: str, port: int, hello: dict, timeout: float, retry_interval: float,
                send_timeout: Optional[float]) -> tuple[TcpLink, dict]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    rejects = 0
    while True:
        attempt = min(CONNECT_ATTEMPT_S, max(deadline - loop.time(), retry_interval))
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), attempt)
        except OSError:  # includes TimeoutError: the attempt itself timed out
            if loop.time() > deadline:
                raise
            await asyncio.sleep(retry_interval)
            continue
        link = TcpLink(reader, writer, send_timeout)
        try:
            await link.send(Frame("hello", hello))
            ack = await link.recv(max(deadline - loop.time(), retry_interval))
            if ack.kind == "hello_ack":
                return link, ack.meta
            # Older peers send an empty meta, so the reason is only for the message.
            why = f"peer rejected hello: {ack.meta.get('reason') or ack.meta}"
        except (LinkClosed, asyncio.TimeoutError) as e:
            why = f"hello failed: {str(e) or 'no reply'}"
        except BaseException:
            # Cancelled mid-hello: no caller holds this connection yet, so nobody else would
            # close it, and the peer may already have taken it as its upstream link.
            link.abort()
            raise
        await link.close()
        if loop.time() > deadline:
            raise LinkClosed(why)
        rejects += 1
        log.log(logging.INFO if rejects == 1 else logging.DEBUG,
                "%s:%s not ready for this stage (%s); retrying", host, port, why)
        await asyncio.sleep(retry_interval)


class LinkServer:
    """Accepts exactly one upstream link whose hello matches ``expect``.

    With ``resume_window`` an upstream that offers a resilient link gets a
    ``ResilientLink``, and later ``resume`` hellos for the same (job, epoch)
    reattach to it instead of being rejected.
    """

    def __init__(self, host: str, port: int, expect: dict,
                 send_timeout: Optional[float] = None,
                 resume_window: Optional[float] = None) -> None:
        self.host, self.port, self.expect = host, port, expect
        self.send_timeout = send_timeout
        self.resume_window = resume_window
        self._resilient: Optional[ResilientLink] = None
        self._accepted: asyncio.Future = asyncio.get_running_loop().create_future()
        self._claimed = False                 # accept() handed the link to the caller
        self._pending: set[TcpLink] = set()   # connections still in the hello exchange
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self) -> "LinkServer":
        # Bound first, served second: asyncio yields once after it starts serving, and a cancel
        # landing there would drop the only reference to a live listener (the port then stays
        # bound for the life of the process).
        self._server = await asyncio.start_server(self._on_conn, self.host, self.port,
                                                  reuse_address=True, start_serving=False)
        try:
            await self._server.start_serving()
        except BaseException:
            self._server.close()
            raise
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    def _reject_reason(self, hello: Frame) -> Optional[str]:
        if hello.kind != "hello":
            return "expected a hello frame"
        wrong = [k for k, v in self.expect.items() if hello.meta.get(k) != v]
        if wrong == ["epoch"]:
            # Same job, so the dialer may learn which epoch this stage is on.
            return f"stage is listening for epoch {self.expect['epoch']}, got {hello.meta.get('epoch')}"
        if wrong:
            return f"hello does not match this stage ({', '.join(wrong)})"
        if "resume" in hello.meta:
            if self._resilient is None or self._resilient.closed:
                return "no link to resume"
            return None
        if self._accepted.done():
            return "stage already has an upstream link"
        return None

    async def _on_conn(self, reader, writer) -> None:
        link = TcpLink(reader, writer, self.send_timeout)
        self._pending.add(link)
        try:
            hello = await link.recv(10.0)
        except Exception:
            await link.close()
            return
        finally:
            self._pending.discard(link)
        reason = self._reject_reason(hello)
        if reason is not None:
            await link.send(Frame("hello_reject", {"reason": reason}))
            await link.close()
            return
        if "resume" in hello.meta:
            resilient = self._resilient
            await link.send(Frame("hello_ack", {"resilient": 1, "recv_seq": resilient.recv_seq}))
            await resilient.attach(link, int(hello.meta["resume"]))
            return
        if hello.meta.get("resilient") and self.resume_window is not None:
            await link.send(Frame("hello_ack", {"resilient": 1}))
            self._resilient = ResilientLink(link, window=self.resume_window)
            self._accepted.set_result(self._resilient)
            return
        await link.send(Frame("hello_ack", {}))
        self._accepted.set_result(link)

    async def accept(self, timeout: Optional[float] = None) -> Link:
        link = await asyncio.wait_for(asyncio.shield(self._accepted), timeout)
        self._claimed = True
        return link

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()                  # stops listening, so the port frees now
        # wait_closed() waits for every connection, so close the ones nobody else
        # will: half-done hellos and a link accepted after the caller gave up on it.
        links = list(self._pending)
        if self._accepted.done() and not self._accepted.cancelled() and not self._claimed:
            links.append(self._accepted.result())
        for link in links:
            await link.close()
        await self._server.wait_closed()
