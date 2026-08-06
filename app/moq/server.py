"""May phat MoQ: QUIC -> HTTP/3 -> WebTransport -> tap con moq-transport.

Duong di cua mot khung hinh:

    camera --RTSP--> engine C++ (giai ma/transcode DUNG CHUNG voi WebRTC)
           --unix socket--> hub.py --> file nay --QUIC--> trinh duyet (WebCodecs)

Mot nguoi xem = mot SUBSCRIBE = mot feed rieng ben engine. Nguon RTSP va bo
transcode thi VAN dung chung — cho tien nam o `CameraSourceRegistry` cua
engine, khong phai o day.

Mot nhom (group) = mot GOP, di tren MOT stream mot chieu rieng va dong lai khi
GOP het. Do la ngu nghia MoQ chuan va no cho mot tinh chat dep: mat mot GOP
khong keo theo GOP sau, vi moi stream QUIC doc lap ve thu tu va truyen lai.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict, Optional

import httpx
from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.h3.connection import H3_ALPN, H3Connection, Setting as H3Setting
from aioquic.h3.events import H3Event, HeadersReceived, WebTransportStreamDataReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import (ConnectionTerminated, ProtocolNegotiated,
                                 QuicEvent, StreamDataReceived, StreamReset)

from app.core.config import settings
from app.moq import cert as cert_mod
from app.moq import wire
from app.moq.hub import Feed, FeedHub

log = logging.getLogger("moq.server")

# Backlog chua duoc bao nhan tren cac stream du lieu. Vuot nguong = nguoi xem
# (hoac duong mang) khong theo kip; luc do bo khung cho toi keyframe ke tiep
# thay vi de bo dem phinh mai. 3 MB ~ hon mot giay o 20 Mbps.
BACKLOG_LIMIT = 3 << 20
# Cho engine noi socket vao sau khi ta bao no tao feed.
FEED_CONNECT_TIMEOUT = 10.0


class _Subscription:
    """Mot track dang phat toi mot nguoi xem."""

    __slots__ = ("subscribe_id", "track_alias", "feed", "group_id", "object_id",
                 "stream_id", "dropping", "session_id", "frames", "bytes")

    def __init__(self, subscribe_id: int, track_alias: int, feed: Feed) -> None:
        self.subscribe_id = subscribe_id
        self.track_alias = track_alias
        self.feed = feed
        self.group_id = 0
        self.object_id = 0
        self.stream_id: Optional[int] = None
        self.dropping = False
        self.session_id = ""
        self.frames = 0
        self.bytes = 0


# SETTINGS_H3_DATAGRAM ban draft-04. aioquic chi gui ban RFC 9297 (0x33), con
# Chrome doi ban nay cho toi khoang phien ban 114 — thieu no thi WebTransport
# chet ngay o "Opening handshake failed" ma khong noi vi sao (do tren chromium
# 110 cua board). Gui CA HAI thi ca trinh duyet cu lan moi deu chay.
H3_DATAGRAM_DRAFT04 = 0xFFD277


class _H3(H3Connection):
    def _get_local_settings(self) -> dict:
        settings = super()._get_local_settings()
        settings[H3_DATAGRAM_DRAFT04] = 1
        return settings

    def _validate_settings(self, settings: dict) -> None:
        # Chrome cu gui ENABLE_WEBTRANSPORT=1 KEM setting datagram ban draft-04,
        # con aioquic chi cong nhan ban RFC (0x33) va dong thang ket noi voi
        # "ENABLE_WEBTRANSPORT requires H3_DATAGRAM". Coi hai ban la mot roi mo
        # cho kiem tra chay tiep — day khong phai noi long an toan gi ca, hai
        # setting nay chi khac nhau ma so.
        if settings.get(H3_DATAGRAM_DRAFT04) == 1:
            settings = dict(settings)
            settings.setdefault(H3Setting.H3_DATAGRAM, 1)
        super()._validate_settings(settings)


class MoqProtocol(QuicConnectionProtocol):
    def __init__(self, *args, hub: FeedHub, engine_url: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = None
        self._hub = hub
        self._engine = engine_url
        self._wt_session: Optional[int] = None
        self._control_stream: Optional[int] = None
        self._control_buf = bytearray()
        self._subs: Dict[int, _Subscription] = {}

    # --- cau noi su kien ---------------------------------------------------
    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ProtocolNegotiated):
            self._http = _H3(self._quic, enable_webtransport=True)
        elif isinstance(event, StreamReset):
            if event.stream_id in (self._control_stream, self._wt_session):
                log.info("[moq] client dong phien (stream %d reset)", event.stream_id)
                self._teardown()
                return
        elif (isinstance(event, StreamDataReceived) and event.end_stream
              and self._wt_session is not None
              and event.stream_id == self._wt_session):
            # `WebTransport.close()` cua trinh duyet KHONG dong ket noi QUIC —
            # o draft-02 (ban Chrome 110 dang dung, xem header
            # sec-webtransport-http3-draft) no dong PHIEN bang cach FIN stream
            # CONNECT. Bat o tang QUIC chu khong doi H3 sinh su kien: H3 coi
            # day la DATA cua mot yeu cau CONNECT va khong bao gi ca.
            #
            # Khong co nhanh nay thi tat camera xong engine VAN bom khung: do
            # duoc /moq/feeds bao 1 nguoi xem suot 20 giay sau khi bam Tắt,
            # va chi nha ra khi dong han tab.
            log.info("[moq] client dong phien (FIN stream CONNECT %d)",
                     event.stream_id)
            self._teardown()
            return
        elif isinstance(event, ConnectionTerminated):
            # PHAI bat o day chu khong dua vao connection_lost(): aioquic chay
            # may chu tren MOT datagram endpoint dung chung, nen connection_lost
            # cua tung QuicConnectionProtocol KHONG bao gio duoc goi (no dung
            # _connection_terminated_handler rieng). Do dung nham cho nay,
            # nguoi xem dong tab ma engine van bom khung mai — feed roi vao
            # khoang khong, khong ai bao ai dung.
            self._teardown()
            return
        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                self._h3_event(h3_event)

    def _h3_event(self, event: H3Event) -> None:
        if isinstance(event, HeadersReceived):
            self._on_headers(event)
        elif isinstance(event, WebTransportStreamDataReceived):
            self._on_wt_data(event)

    def _on_headers(self, event: HeadersReceived) -> None:
        headers = {k: v for k, v in event.headers}
        method = headers.get(b":method", b"")
        protocol = headers.get(b":protocol", b"")
        path = headers.get(b":path", b"")
        if method != b"CONNECT" or protocol != b"webtransport" or path != b"/moq":
            self._http.send_headers(
                stream_id=event.stream_id,
                headers=[(b":status", b"404"), (b"server", b"vms-moq")],
                end_stream=True,
            )
            self.transmit()
            return
        self._wt_session = event.stream_id
        self._http.send_headers(
            stream_id=event.stream_id,
            headers=[
                (b":status", b"200"),
                (b"sec-webtransport-http3-draft", b"draft02"),
                (b"server", b"vms-moq"),
            ],
        )
        self.transmit()
        log.info("[moq] phien WebTransport mo (stream %d)", event.stream_id)

    def _on_wt_data(self, event: WebTransportStreamDataReceived) -> None:
        # Stream hai chieu DAU TIEN client mo chinh la control stream cua MoQ.
        if self._control_stream is None:
            self._control_stream = event.stream_id
        if event.stream_id != self._control_stream:
            return
        self._control_buf += event.data
        self._drain_control()
        if event.stream_ended:
            self._teardown()

    def _drain_control(self) -> None:
        reader = wire.Reader(bytes(self._control_buf))
        while True:
            try:
                msg_type, body = wire.read_control(reader)
            except wire.Incomplete:
                break
            try:
                self._on_control(msg_type, body)
            except Exception:
                log.exception("[moq] ban tin control 0x%x hong", msg_type)
        del self._control_buf[: reader.pos]

    def _on_control(self, msg_type: int, body: wire.Reader) -> None:
        if msg_type == wire.CLIENT_SETUP:
            versions = [body.varint() for _ in range(body.varint())]
            if wire.VERSION not in versions:
                log.warning("[moq] client doi phien ban %s, ta chi co 0x%x",
                            [hex(v) for v in versions], wire.VERSION)
                self.close()
                return
            self._send_control(wire.server_setup())
        elif msg_type == wire.SUBSCRIBE:
            asyncio.ensure_future(self._on_subscribe(body))
        elif msg_type == wire.UNSUBSCRIBE:
            self._drop_sub(body.varint())

    # --- dang ky track -----------------------------------------------------
    async def _on_subscribe(self, body: wire.Reader) -> None:
        subscribe_id = body.varint()
        track_alias = body.varint()
        namespace = body.tuple_()
        body.string()          # track name — ta chi co mot track video moi phien
        body.u8()              # publisher priority
        body.u8()              # group order
        body.varint()          # filter type
        params = body.params()

        # namespace = ["vms", "live"|"playback", <cameraId>]
        if len(namespace) != 3 or namespace[0] != "vms":
            self._send_control(wire.subscribe_error(
                subscribe_id, wire.ERR_NOT_FOUND, "namespace la"))
            return
        mode, camera_id = namespace[1], namespace[2]
        if mode not in ("live", "playback"):
            self._send_control(wire.subscribe_error(
                subscribe_id, wire.ERR_NOT_FOUND, "mode la"))
            return

        start_ms = int.from_bytes(params.get(wire.PARAM_START_MS, b"\x00"), "big")
        rate_milli = int.from_bytes(params.get(wire.PARAM_RATE_MILLI, b""), "big") or 1000

        feed_id = uuid.uuid4().hex
        feed = self._hub.create(feed_id)
        sub = _Subscription(subscribe_id, track_alias, feed)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self._engine}/moq/feeds",
                    json={
                        "feed": feed_id,
                        "cameraId": camera_id,
                        "mode": mode,
                        "atMs": start_ms,
                        "rate": rate_milli / 1000.0,
                    },
                )
            if resp.status_code >= 300:
                raise RuntimeError(f"engine {resp.status_code}: {resp.text[:200]}")
            sub.session_id = resp.json().get("sessionId", "")
        except Exception as exc:
            log.error("[moq] engine tu choi feed %s: %s", camera_id, exc)
            self._hub.drop(feed_id)
            self._send_control(wire.subscribe_error(
                subscribe_id, wire.ERR_INTERNAL, str(exc)[:120]))
            return

        try:
            await asyncio.wait_for(feed.connected.wait(), FEED_CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            log.error("[moq] engine khong noi vao feed %s sau %.0fs",
                      feed_id, FEED_CONNECT_TIMEOUT)
            await self._kill_feed(feed_id, sub.session_id)
            self._send_control(wire.subscribe_error(
                subscribe_id, wire.ERR_INTERNAL, "engine khong bom khung"))
            return

        self._subs[subscribe_id] = sub
        feed.on_frame = lambda pts, key, au: self._on_frame(sub, pts, key, au)
        self._send_control(wire.subscribe_ok(subscribe_id, sub.session_id))
        log.info("[moq] SUBSCRIBE %s/%s -> phien %s", mode, camera_id, sub.session_id)

    # --- phat du lieu ------------------------------------------------------
    def _on_frame(self, sub: _Subscription, pts_us: int, keyframe: bool,
                  au: bytes) -> None:
        if self._quic is None:
            return
        if keyframe:
            # Ket thuc GOP cu roi mo stream moi. Dong stream la tin hieu
            # "nhom nay het" cho ben nhan, khong can co rieng.
            if sub.stream_id is not None:
                self._quic.send_stream_data(sub.stream_id, b"", end_stream=True)
            sub.stream_id = self._http.create_webtransport_stream(
                session_id=self._wt_session, is_unidirectional=True)
            self._quic.send_stream_data(
                sub.stream_id, wire.subgroup_header(sub.track_alias, sub.group_id))
            sub.group_id += 1
            sub.object_id = 0
            sub.dropping = False
        elif sub.stream_id is None or sub.dropping:
            # Chua co keyframe nao, hoac dang bo khung cho keyframe ke tiep:
            # bom P-frame vao luc nay chi ra hinh vo toi IDR sau.
            return

        if self._backlog() > BACKLOG_LIMIT:
            if not sub.dropping:
                log.warning("[moq] phien %s cham, bo khung toi keyframe ke tiep",
                            sub.session_id)
            sub.dropping = True
            return

        payload = wire.pack_object_payload(pts_us, keyframe, au)
        self._quic.send_stream_data(sub.stream_id, wire.obj(sub.object_id, payload))
        sub.object_id += 1
        sub.frames += 1
        sub.bytes += len(au)
        self.transmit()

    def _backlog(self) -> int:
        """Byte da ghi ma chua duoc bao nhan, tren moi stream cua ket noi nay.

        Dung thuoc tinh rieng cua aioquic (`_streams`, `sender`) — co fallback
        tra 0 de mot lan aioquic doi ten khong lam chet duong xem, chi mat co
        che chong phinh bo dem.
        """
        try:
            total = 0
            for st in self._quic._streams.values():
                sender = getattr(st, "sender", None)
                if sender is None or sender.is_finished:
                    continue
                total += sender.highest_offset - getattr(sender, "_buffer_start", 0)
            return total
        except Exception:
            return 0

    # --- don dep -----------------------------------------------------------
    def _drop_sub(self, subscribe_id: int) -> None:
        sub = self._subs.pop(subscribe_id, None)
        if sub is None:
            return
        sub.feed.on_frame = None
        if sub.stream_id is not None and self._quic is not None:
            try:
                self._quic.send_stream_data(sub.stream_id, b"", end_stream=True)
                self.transmit()
            except Exception:
                pass
        log.info("[moq] dong phien %s (%d khung, %.1f MB)",
                 sub.session_id, sub.frames, sub.bytes / 1e6)
        asyncio.ensure_future(self._kill_feed(sub.feed.feed_id, sub.session_id))

    async def _kill_feed(self, feed_id: str, session_id: str) -> None:
        self._hub.drop(feed_id)
        if not session_id:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.delete(f"{self._engine}/moq/feeds/{session_id}")
        except Exception as exc:
            log.warning("[moq] khong xoa duoc phien engine %s: %s", session_id, exc)

    def _teardown(self) -> None:
        for subscribe_id in list(self._subs):
            self._drop_sub(subscribe_id)

    def _send_control(self, data: bytes) -> None:
        if self._control_stream is None or self._quic is None:
            return
        self._quic.send_stream_data(self._control_stream, data)
        self.transmit()


class MoqServer:
    """Vong doi may chu, chay tren event loop RIENG trong mot thread daemon.

    Khong dung chung loop voi uvicorn: mot ket noi QUIC bi cham se lam cham ca
    REST/WebSocket cua he, ma day chi la duong xem phu.
    """

    def __init__(self) -> None:
        self.info: dict = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._hub: Optional[FeedHub] = None
        self._stop: Optional[asyncio.Event] = None

    def worker(self) -> None:
        if not settings.MOQ_ENABLED:
            log.info("[moq] tat theo cau hinh")
            return
        try:
            asyncio.run(self._run())
        except Exception:
            log.exception("[moq] may chu dung han")

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()

        material = cert_mod.ensure(settings.MOQ_CERT_DIR)
        self.info = {
            "port": settings.MOQ_PORT,
            "path": "/moq",
            "fingerprint": material["fingerprint"],
            "expiresAt": material["expires_at"],
        }

        self._hub = FeedHub(settings.MOQ_FEED_SOCKET)
        await self._hub.start()

        config = QuicConfiguration(
            alpn_protocols=H3_ALPN,
            is_client=False,
            # WebTransport tren HTTP/3 doi SETTINGS_H3_DATAGRAM, ma aioquic chi
            # gui setting do khi datagram duoc bat — khong dat thi Chrome tu
            # choi phien ngay sau CONNECT.
            max_datagram_frame_size=65536,
            # Video 1080p day nhieu byte hon mac dinh cua aioquic rat nhieu;
            # de nguyen thi cua so luong bi khoa lien tuc.
            max_data=64 << 20,
            max_stream_data=16 << 20,
        )
        config.load_cert_chain(material["cert_file"], material["key_file"])

        hub, engine = self._hub, settings.AI_API_BASE_URL

        await serve(
            settings.MOQ_HOST,
            settings.MOQ_PORT,
            configuration=config,
            create_protocol=lambda *a, **kw: MoqProtocol(
                *a, hub=hub, engine_url=engine, **kw),
        )
        log.info("[moq] nghe QUIC tai %s:%d, van tay %s",
                 settings.MOQ_HOST, settings.MOQ_PORT, material["fingerprint"][:16])
        await self._stop.wait()
        await self._hub.stop()

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)


moq_server = MoqServer()
