"""Duong nhan access unit tu engine C++ qua unix socket.

VI SAO KHONG DE ENGINE TU NOI QUIC: khong co thu vien QUIC nao trong ban dung
C++ tren board, ma keo msquic/quiche vao la them mot phu thuoc lon cho mot
duong xem phu. Phia Python thi aioquic co san banh xe (H3 + WebTransport) va
da duoc kiem chung. Chi phi phai tra la mot lan chep qua unix socket: o
2-4 Mbps mot camera thi khong dang ke so voi giai ma/transcode ma engine da
lam — va phan dat tien do VAN dung chung voi WebRTC, khong nhan doi.

Moi phien xem = mot ket noi socket rieng (mot "feed"). Engine goi den, khai
bao minh la feed nao, roi bom access unit.

Khung tren day, tat ca so BE:
    tieu de : b"MOQF1 " + JSON mot dong + b"\\n"
    moi khung: u8 co | u64 pts_us | u32 do_dai | Annex-B
    co bit0 = keyframe
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable, Dict, Optional

log = logging.getLogger("moq.hub")

MAGIC = b"MOQF1 "
FLAG_KEYFRAME = 0x01
# Mot access unit 1080p keyframe co the vai tram KB; 8 MB la tran chong ban
# tin hong lam cap phat mot mieng khong lo.
MAX_AU_BYTES = 8 << 20

FrameCb = Callable[[int, bool, bytes], None]


class Feed:
    """Mot phien bom khung. Tao TRUOC khi bao engine ket noi vao."""

    def __init__(self, feed_id: str) -> None:
        self.feed_id = feed_id
        self.on_frame: Optional[FrameCb] = None
        self.session_id: str = ""      # id phien ben engine (de dieu khien xem lai)
        self.codec: str = "h264"
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()
        self._writer: Optional[asyncio.StreamWriter] = None

    def close(self) -> None:
        self.closed.set()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None


class FeedHub:
    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self._feeds: Dict[str, Feed] = {}
        self._server: Optional[asyncio.AbstractServer] = None

    # --- vong doi ---------------------------------------------------------
    async def start(self) -> None:
        # Socket cu con lai sau khi tien trinh bi kill: bind se bao "Address
        # already in use" du khong ai nghe. Xoa truoc.
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        self._server = await asyncio.start_unix_server(self._serve, self.socket_path)
        os.chmod(self.socket_path, 0o666)
        log.info("[moq] nhan khung tai %s", self.socket_path)

    async def stop(self) -> None:
        for feed in list(self._feeds.values()):
            feed.close()
        self._feeds.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # --- dang ky ----------------------------------------------------------
    def create(self, feed_id: str) -> Feed:
        feed = Feed(feed_id)
        self._feeds[feed_id] = feed
        return feed

    def drop(self, feed_id: str) -> None:
        feed = self._feeds.pop(feed_id, None)
        if feed is not None:
            feed.close()

    # --- doc ---------------------------------------------------------------
    async def _serve(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        feed: Optional[Feed] = None
        try:
            magic = await reader.readexactly(len(MAGIC))
            if magic != MAGIC:
                log.warning("[moq] feed sai magic %r", magic)
                return
            header = json.loads((await reader.readline()).decode())
            feed = self._feeds.get(header.get("feed", ""))
            if feed is None:
                # Nguoi xem da bo di truoc khi engine kip noi vao.
                log.info("[moq] feed %s khong con ai doi", header.get("feed"))
                return
            feed.codec = header.get("codec", "h264")
            feed._writer = writer
            feed.connected.set()
            log.info("[moq] feed %s da noi (%s)", feed.feed_id, feed.codec)
            await self._pump(reader, feed)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception:
            log.exception("[moq] loi doc feed")
        finally:
            if feed is not None:
                feed.closed.set()
            try:
                writer.close()
            except Exception:
                pass

    @staticmethod
    async def _pump(reader: asyncio.StreamReader, feed: Feed) -> None:
        while not feed.closed.is_set():
            head = await reader.readexactly(13)
            flags = head[0]
            pts_us = int.from_bytes(head[1:9], "big")
            length = int.from_bytes(head[9:13], "big")
            if length > MAX_AU_BYTES:
                log.error("[moq] feed %s bao do dai vo ly %d", feed.feed_id, length)
                return
            au = await reader.readexactly(length)
            cb = feed.on_frame
            if cb is not None:
                cb(pts_us, bool(flags & FLAG_KEYFRAME), au)
