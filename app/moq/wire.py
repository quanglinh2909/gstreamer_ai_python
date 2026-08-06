"""Ma hoa/giai ma khung day cua MoQ (Media over QUIC).

Day la mot TAP CON cua draft-ietf-moq-transport, du de mot may phat va mot
trinh duyet noi chuyen voi nhau, KHONG phai ban day du:

  co        : varint kieu QUIC (RFC 9000 muc 16), control stream, SETUP,
              SUBSCRIBE / SUBSCRIBE_OK / SUBSCRIBE_ERROR / UNSUBSCRIBE,
              du lieu di theo tung nhom tren stream mot chieu (SUBGROUP).
  khong co  : ANNOUNCE / relay / cache / FETCH / uu tien nhom / track audio,
              va cac tham so SETUP ngoai MAX_SUBSCRIBE_ID.

Vi sao lam tap con thay vi lay nguyen mot thu vien: ban ve van dang doi giua
cac draft (draft-06 khong co do dai cho ban tin control, draft-07 co), moi
hien thuc bam mot moc khac nhau. Ta so huu CA HAI DAU nen chot mot ban va ghi
ro o day; doi lay viec khong phai keo Rust toolchain len board va khong bi
mot ban nang cap draft lam gay san pham dang chay.

Ban tin control CO do dai (varint) o day — theo huong draft-07 tro di, vi
khong co do dai thi mot kieu ban tin la se lam hong ca dong.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# --- kieu ban tin control (bam theo draft-ietf-moq-transport) ---------------
CLIENT_SETUP = 0x40
SERVER_SETUP = 0x41
SUBSCRIBE = 0x03
SUBSCRIBE_OK = 0x04
SUBSCRIBE_ERROR = 0x05
UNSUBSCRIBE = 0x0A

# Kieu header cua stream du lieu mot chieu.
SUBGROUP_HEADER = 0x04

# Phien ban ta chot. Draft dung khong gian 0xff00000X; ta them mot so rieng
# de mot client MoQ that (moq-rs...) tu choi ngay tu SETUP thay vi noi chuyen
# nua voi dong container rieng cua ta roi ra hinh vo.
VERSION = 0xFF00000B

# Tham so SUBSCRIBE rieng cua ta (khong gian tren 0x1000 de khong dam vao
# cac tham so draft dinh nghia o day duoi).
PARAM_START_MS = 0x1000   # xem lai: moc thoi gian epoch-ms bat dau
PARAM_RATE_MILLI = 0x1001  # xem lai: toc do x1000 (1000 = 1.0x)
PARAM_SESSION_ID = 0x1002  # SUBSCRIBE_OK tra ve: id phien ben engine

# Loi
ERR_NOT_FOUND = 0x04
ERR_INTERNAL = 0x05

# Co trong payload cua doi tuong (xem `pack_object_payload`).
FLAG_KEYFRAME = 0x01


class Reader:
    """Doc tuan tu mot bytes-like. Thieu byte thi nem `Incomplete`."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def take(self, n: int) -> bytes:
        if self.remaining() < n:
            raise Incomplete()
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def varint(self) -> int:
        if self.remaining() < 1:
            raise Incomplete()
        first = self.data[self.pos]
        length = 1 << (first >> 6)
        if self.remaining() < length:
            raise Incomplete()
        value = first & 0x3F
        for i in range(1, length):
            value = (value << 8) | self.data[self.pos + i]
        self.pos += length
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def blob(self) -> bytes:
        return self.take(self.varint())

    def string(self) -> str:
        return self.blob().decode("utf-8", "replace")

    def tuple_(self) -> List[str]:
        return [self.string() for _ in range(self.varint())]

    def params(self) -> dict:
        out = {}
        for _ in range(self.varint()):
            key = self.varint()
            out[key] = self.blob()
        return out


class Incomplete(Exception):
    """Chua du byte — goi lai khi co them du lieu."""


def varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint am")
    if value < 0x40:
        return bytes([value])
    if value < 0x4000:
        return (value | 0x4000).to_bytes(2, "big")
    if value < 0x4000_0000:
        return (value | 0x8000_0000).to_bytes(4, "big")
    if value < 0x4000_0000_0000_0000:
        return (value | 0xC000_0000_0000_0000).to_bytes(8, "big")
    raise ValueError("varint qua lon")


def blob(data: bytes) -> bytes:
    return varint(len(data)) + data


def string(text: str) -> bytes:
    return blob(text.encode("utf-8"))


def tuple_(parts: Sequence[str]) -> bytes:
    return varint(len(parts)) + b"".join(string(p) for p in parts)


def params(items: dict) -> bytes:
    out = varint(len(items))
    for key, value in items.items():
        out += varint(key) + blob(value)
    return out


def control(msg_type: int, payload: bytes) -> bytes:
    """Bo ban tin control: kieu + do dai + noi dung."""
    return varint(msg_type) + varint(len(payload)) + payload


def read_control(reader: Reader) -> Tuple[int, Reader]:
    """Doc mot ban tin control tron ven. Nem `Incomplete` neu chua du."""
    start = reader.pos
    try:
        msg_type = reader.varint()
        length = reader.varint()
        body = reader.take(length)
    except Incomplete:
        reader.pos = start  # tra con tro ve de lan sau doc lai tu dau
        raise
    return msg_type, Reader(body)


# --- phia may phat ---------------------------------------------------------

def server_setup() -> bytes:
    return control(SERVER_SETUP, varint(VERSION) + varint(0))


def subscribe_ok(subscribe_id: int, session_id: str) -> bytes:
    body = (
        varint(subscribe_id)
        + varint(0)          # expires = 0: khong het han
        + bytes([0x01])      # group order: tang dan
        + bytes([0x00])      # contentExists = 0 (luon phat tu nhom ke tiep)
        + params({PARAM_SESSION_ID: session_id.encode()})
    )
    return control(SUBSCRIBE_OK, body)


def subscribe_error(subscribe_id: int, code: int, reason: str) -> bytes:
    body = varint(subscribe_id) + varint(code) + string(reason) + varint(0)
    return control(SUBSCRIBE_ERROR, body)


def subgroup_header(track_alias: int, group_id: int, priority: int = 0x80) -> bytes:
    return (
        varint(SUBGROUP_HEADER)
        + varint(track_alias)
        + varint(group_id)
        + varint(0)          # subgroup id: ta chi dung mot subgroup moi nhom
        + bytes([priority])
    )


def obj(object_id: int, payload: bytes) -> bytes:
    return varint(object_id) + blob(payload)


def pack_object_payload(pts_us: int, keyframe: bool, au: bytes) -> bytes:
    """Container trong mot doi tuong MoQ: 1 byte co + 8 byte PTS + Annex-B.

    Vi sao Annex-B chu khong phai AVCC: WebCodecs coi bitstream la Annex-B khi
    `VideoDecoder.configure` KHONG co truong `description`. Nho vay trinh phat
    khong can dung avcC, va engine gui thang caps `stream-format=byte-stream`
    ma CameraRtpSource von da tao san — khong them mot lan chuyen dinh dang.
    """
    flags = FLAG_KEYFRAME if keyframe else 0
    return bytes([flags]) + pts_us.to_bytes(8, "big") + au
