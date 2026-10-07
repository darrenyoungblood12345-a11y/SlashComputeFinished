"""LAN discovery of the coordinator over mDNS."""

from __future__ import annotations

import queue
import socket
import time
from typing import Callable, Optional

from slashcompute.common.config import MDNS_SERVICE_TYPE


def lan_ip() -> str:
    """Best-effort primary LAN address (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Advertiser:
    """Advertises the coordinator on the LAN from inside its running event loop.

    zeroconf's blocking API raises ``EventLoopBlocked`` when called from the loop it runs on,
    so registration goes through ``AsyncZeroconf``.
    """

    def __init__(self, port: int, host_ip: Optional[str] = None) -> None:
        from zeroconf import ServiceInfo

        ip = host_ip or lan_ip()
        name = f"slashcompute-{socket.gethostname().split('.')[0]}"
        self._info = ServiceInfo(
            MDNS_SERVICE_TYPE, f"{name}.{MDNS_SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)], port=port, properties={"path": "/ws/agent"},
        )
        self._azc = None

    @property
    def name(self) -> str:
        return self._info.name

    async def start(self) -> None:
        from zeroconf.asyncio import AsyncZeroconf

        self._azc = AsyncZeroconf()
        try:
            # Another coordinator on the LAN may use this Mac's name: take "name (2)" instead.
            await (await self._azc.async_register_service(self._info, allow_name_change=True))
        except BaseException:
            await self._azc.async_close()
            self._azc = None
            raise

    async def close(self) -> None:
        if self._azc is None:
            return
        await self._azc.async_unregister_service(self._info)
        await self._azc.async_close()
        self._azc = None


def discover(timeout: float = 5.0, exclude: Optional[Callable[[str], bool]] = None) -> Optional[str]:
    """Return ``http://host:port`` of the first coordinator advertised on the LAN that ``exclude``
    does not reject (any, by default), or None once ``timeout`` has passed. Advertisements are
    taken as they arrive until then: while hosting, the first one is this Mac's own pool."""
    from zeroconf import ServiceBrowser, Zeroconf

    found: queue.Queue[str] = queue.Queue()   # filled from zeroconf's thread

    class _Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=2000)
            if info and info.addresses:
                found.put(f"http://{socket.inet_ntoa(info.addresses[0])}:{info.port}")

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, MDNS_SERVICE_TYPE, _Listener())
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                url = found.get(timeout=remaining)
            except queue.Empty:
                break
            if exclude is None or not exclude(url):
                return url
    finally:
        zc.close()
    return None
