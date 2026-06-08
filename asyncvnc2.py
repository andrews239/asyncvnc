from asyncio import StreamReader, StreamWriter, open_connection, IncompleteReadError, sleep
from contextlib import asynccontextmanager, contextmanager, ExitStack
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import product
from os import urandom
from typing import Callable, Dict, List, Optional, Set, Tuple
from zlib import decompressobj

import numpy as np

from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import load_der_public_key

from keysymdef import keysymdef  # type: ignore


# Keyboard keys
key_codes: Dict[str, int] = {}
key_codes.update((name, code) for name, code, char in keysymdef)
key_codes.update((chr(char), code) for name, code, char in keysymdef if char)
key_codes['Del'] = key_codes['Delete']
key_codes['Esc'] = key_codes['Escape']
key_codes['Cmd'] = key_codes['Super_L']
key_codes['Alt'] = key_codes['Alt_L']
key_codes['Ctrl'] = key_codes['Control_L']
key_codes['Super'] = key_codes['Super_L']
key_codes['Shift'] = key_codes['Shift_L']
key_codes['Backspace'] = key_codes['BackSpace']

# Common screen aspect ratios
screen_ratios: Set[Fraction] = {
    Fraction(3, 2), Fraction(4, 3), Fraction(16, 10), Fraction(16, 9), Fraction(32, 9), Fraction(64, 27)}

# Colour channel orders
video_modes: Dict[bytes, str] = {
     b'\x20\x18\x00\x01\x00\xff\x00\xff\x00\xff\x10\x08\x00': 'bgra',
     b'\x20\x18\x00\x01\x00\xff\x00\xff\x00\xff\x00\x08\x10': 'rgba',
     b'\x20\x18\x01\x01\x00\xff\x00\xff\x00\xff\x10\x08\x00': 'argb',
     b'\x20\x18\x01\x01\x00\xff\x00\xff\x00\xff\x00\x08\x10': 'abgr',
}

video_definition: Dict[str, bytes] = {v: k for k, v in video_modes.items()}

class Enc(Enum):
    """
    Supported encodings

    Default priority order is ZRLE, TRLE, ZLIB, COPY, RAW.

    Pseudo-encodings (negative values per the RFB spec) carry server
    notifications rather than pixel data and are not advertised by default
    in SetEncodings -- the dispatch in Video.read still handles them if a
    server pushes one unsolicited (e.g. on a server-side geometry change).
    """

    #: ZRLE encoding.
    ZRLE = 16

    #: TRLE encoding.
    TRLE = 15

    #: Raw encoding with zlib compession.
    ZLIB = 6

    #: CopyRect encoding.
    COPY = 1

    #: Raw encoding.
    RAW = 0

    #: DesktopSize pseudo-encoding (RFB §7.7.2). Server-side geometry
    #: change; rect.width/height carry the new dimensions, no payload.
    DESKTOP_SIZE = -223

    #: ExtendedDesktopSize pseudo-encoding (libvncserver/RealVNC ext).
    #: Same intent as DESKTOP_SIZE plus a multi-screen layout payload.
    EXTENDED_DESKTOP_SIZE = -308

    @classmethod
    def default(cls):
        """
        Get the default list of supported encodings.

        Excludes RAW (value 0) and pseudo-encodings (negative). Callers
        wanting resize notifications can opt in explicitly, e.g.
        ``EncList() + Enc.EXTENDED_DESKTOP_SIZE``, but be aware that
        TigerVNC treats EDS in SetEncodings as a subscription that returns
        EDS-only FBU responses, starving the screenshot path of pixel data.
        """
        return filter(lambda x: x.value > 0, cls)

class EncList(list):
    """
    A list of Supported encodings and the simple operations:

    "-" -- Exclude from the list

    "+" -- Add to list or rise priority

    EncList()         -- default encoding list

    EncList(Enc.ZLIB) -- encoding list with one encoding

    EncList([Enc.ZLIB,Enc.TRLE]) -- encoding list with two encodings

    EncList() - Enc.ZLIB -- default encoding list except Enc.ZLIB

    EncList() + Enc.ZLIB -- default encoding list with hi-priority of Enc.ZLIB
    """

    def __init__(self, iterable=None):
        if iterable is None:
            super().__init__(Enc.default())
            return
        super().__init__([])
        if isinstance(iterable, Enc):
            iterable = [ iterable ]
        try:
            for x in iterable:
                if isinstance(x, Enc):
                    if not x in self:
                        self.append(x)
                else:
                    raise TypeError
        except TypeError:
            raise TypeError("Arg must be one of the None or the Enc or the iterable of Enc")

    def __add__(self, other):
        if isinstance(other, Enc):
            return EncList([other, *self])
        return EncList([*other, *self])

    def __sub__(self, other):
        if isinstance(other, Enc):
            other = [other]
        a = filter(lambda x: not x in other, self)

        return EncList(a)


async def read_int(reader: StreamReader, length: int) -> int:
    """
    Reads, unpacks, and returns an integer of *length* bytes.
    """

    return int.from_bytes(await reader.readexactly(length), 'big')


async def read_text(reader: StreamReader, encoding: str) -> str:
    """
    Reads, unpacks, and returns length-prefixed text.
    """

    length = await read_int(reader, 4)
    data = await reader.readexactly(length)
    return data.decode(encoding)


def pack_ard(data):
    data = data.encode('utf-8') + b'\x00'
    if len(data) < 64:
        data += urandom(64 - len(data))
    else:
        data = data[:64]
    return data


@dataclass
class Clipboard:
    """
    Shared clipboard.
    """

    writer: StreamWriter = field(repr=False)

    #: The clipboard text.
    text: str = ''

    def write(self, text: str):
        """
        Sends clipboard text to the server.
        """

        data = text.encode('latin-1')
        self.writer.write(b'\x06\x00\x00\x00' + len(data).to_bytes(4, 'big') + data)


@dataclass
class Keyboard:
    """
    Virtual keyboard.
    """

    writer: StreamWriter = field(repr=False)

    @contextmanager
    def _write(self, key: str):
        data = key_codes[key].to_bytes(4, 'big')
        self.writer.write(b'\x04\x01\x00\x00' + data)
        try:
            yield
        finally:
            self.writer.write(b'\x04\x00\x00\x00' + data)

    @contextmanager
    def hold(self, *keys: str):
        """
        Context manager that pushes the given keys on enter, and releases them (in reverse order) on exit.
        """

        with ExitStack() as stack:
            for key in keys:
                stack.enter_context(self._write(key))
            yield

    def press(self, *keys: str):
        """
        Pushes all the given keys, and then releases them in reverse order.
        """

        with self.hold(*keys):
            pass

    def write(self, text: str):
        """
        Pushes and releases each of the given keys, one after the other.
        """

        for key in text:
            with self.hold(key):
                pass


@dataclass
class Mouse:
    """
    Virtual mouse.
    """

    writer: StreamWriter = field(repr=False)
    buttons: int = 0
    x: int = 0
    y: int = 0

    def _write(self):
        self.writer.write(
            b'\x05' +
            self.buttons.to_bytes(1, 'big') +
            self.x.to_bytes(2, 'big') +
            self.y.to_bytes(2, 'big'))

    @contextmanager
    def hold(self, button: int = 0):
        """
        Context manager that presses a mouse button on enter, and releases it on exit.
        """

        mask = 1 << button
        self.buttons |= mask
        self._write()
        try:
            yield
        finally:
            self.buttons &= ~mask
            self._write()

    def click(self, button: int = 0):
        """
        Presses and releases a mouse button.
        """

        with self.hold(button):
            pass

    def middle_click(self):
        """
        Presses and releases the middle mouse button.
        """

        self.click(1)

    def right_click(self):
        """
        Presses and releases the right mouse button.
        """

        self.click(2)

    def scroll_up(self, repeat=1):
        """
        Scrolls the mouse wheel upwards.
        """

        for _ in range(repeat):
            self.click(3)

    def scroll_down(self, repeat=1):
        """
        Scrolls the mouse wheel downwards.
        """

        for _ in range(repeat):
            self.click(4)

    def move(self, x: int, y: int):
        """
        Moves the mouse cursor to the given co-ordinates.
        """

        self.x = x
        self.y = y
        self._write()


@dataclass
class Screen:
    """
    Computer screen.
    """

    #: Horizontal position in pixels.
    x: int

    #: Vertical position in pixels.
    y: int

    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    @property
    def slices(self) -> Tuple[slice, slice]:
        """
        Object that can be used to crop the video buffer to this screen.
        """

        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    @property
    def score(self) -> float:
        """
        A measure of our confidence that this represents a real screen. For screens with standard aspect ratios, this
        is proportional to its pixel area. For non-standard aspect ratios, the score is further multiplied by the ratio
        or its reciprocal, whichever is smaller.
        """

        value = float(self.width * self.height)
        ratios = {Fraction(self.width, self.height).limit_denominator(64),
                  Fraction(self.height, self.width).limit_denominator(64)}
        if not ratios & screen_ratios:
            value *= min(ratios) * 0.5
        return value





@dataclass
class StreamZReader:
    """
    aio StreamReader wrapper for zlib
    """
    reader: StreamReader = field(repr=False)
    decompress: object = field(repr=False)
    length: int
    _head: int = 0
    _buffer: bytes = b''
    _zbuffer: bytes = b''

    async def read(self, n: int =-1) -> bytes:
        """
        Read up to a maximum of n bytes.
        If n is not provided, or set to -1, read until EOF and return all read bytes.
        When n is provided, data will be returned as soon as it is available.
        """

        if n == -1:
            try:
                self._zbuffer += await self.reader.readexactly(self.length)
            except IncompleteReadError as e:
                self._zbuffer += e.partial
            self.length = 0
            rdata = self._buffer[self._head:] + self.decompress.decompress(self._zbuffer)
            self._buffer = b''
            self._head = 0
            self._zbuffer = self.decompress.unconsumed_tail
            return rdata

        if (n <= 0) or ((len(self._buffer) <= self._head) and (self.length <= 0)):
            return b''

        while (len(self._buffer) == self._head) and (self.length > 0):
            ndata = await self.reader.readexactly(self.length)
            self.length -= len(ndata)
            self._zbuffer += ndata
            self._buffer = self.decompress.decompress(self._zbuffer)
            self._head = 0
            self._zbuffer = self.decompress.unconsumed_tail
        rdata = self._buffer[self._head:self._head + n]
        self._head += len(rdata)
        return rdata

    async def readexactly(self, n: int) -> bytes:
        """
        Read exactly n bytes.
        Raise an asyncio.IncompleteReadError if the end of the stream is reached before n can be read.
        """
        data = b''
        _n = n
        while _n > 0:
            ndata = await self.read(_n)
            if len(ndata) == 0:
                raise IncompleteReadError(data, n)
            data += ndata
            _n = n - len(data)
        return data


def _tile_1d_gen(tw: int, x: int, w: int):
    for cx in range(x, x + w - tw + 1, tw):
        yield (cx, tw)
    mod = w % tw
    if mod:
        yield (x + w - mod, mod)

def _tile_gen(tw: int, th: int, x: int, y: int, w: int, h: int):
    for (cy, ch) in _tile_1d_gen(th, y, h):
        for (cx, cw) in _tile_1d_gen(tw, x, w):
            yield (cx, cy, cw, ch)

async def _rle_len(reader: StreamReader):
    _pixels = 0
    while True:
        n = await read_int(reader, 1)
        _pixels += n
        if n != 255:
            break
    return _pixels + 1

async def _update_palette(reader: StreamReader, n: int, palette: list = None):
    pal_bytes = await reader.readexactly(3*n)
    pal_list = list(pal_bytes[i:i+3] for i in range(0, 3*n, 3))
    palette.clear()
    palette.extend(pal_list)

async def _rle_packedbits(reader: StreamReader, cw: int, ch: int, subencoding: int, palette: list) -> bytes:
    frame = b''
    if subencoding <= 16:
        # subencoding == 2 .. 16
        await _update_palette(reader, subencoding, palette)
    else:
        # subencoding == 127
        subencoding = len(palette)

    if subencoding == 2:
        bits = 1
    elif subencoding <= 4:
        bits = 2
    else:
        bits = 4
    rowlen = (bits * cw + 7) // 8
    mask = (1 << bits) - 1

    for _ in range(ch):
        row = await reader.readexactly(rowlen)
        offset = 0
        rowit = iter(row)
        for _ in range(cw):
            if offset == 0:
                offset = 8
                packcol = next(rowit)
            offset -= bits
            frame += palette[(packcol >> offset) & mask]
    return frame

async def _rle_rle(reader: StreamReader, cw: int, ch: int, subencoding: int, palette: list) -> bytes:
    frame = b''
    subencoding -= 128
    if subencoding > 1:
        await _update_palette(reader, subencoding, palette)
    elif subencoding == 1:
        subencoding = len(palette)

    tile_pixels = ch * cw
    pixels = 0

    while pixels < tile_pixels:
        if subencoding:
            pal_index = await read_int(reader, 1)
            if pal_index < 128:
                frame += palette[pal_index]
                pixels += 1
                continue
            else:
                pal_index -= 128
                color = palette[pal_index]
        else:
            color = await reader.readexactly(3)
        p = await _rle_len(reader)
        frame += color * p
        pixels += p
    if pixels > tile_pixels:
        raise ValueError("Too many pixels")
    return frame


@dataclass
class Video:
    """
    Video buffer.
    """

    reader: StreamReader = field(repr=False)
    writer: StreamWriter = field(repr=False)
    decompress: object = field(repr=False)
#    lastrefresh = None;

    #: Desktop name.
    name: str

    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    #: Colour channel order.
    mode: str

    #: Allowed encodings list
    encodings: Optional[list] = None

    #: Serial number
    serial = 0

    #: 3D numpy array of colour data.
    data: Optional[np.ndarray] = None

    @classmethod
    async def create(cls, reader: StreamReader, writer: StreamWriter, encodings: list) -> 'Video':
        encodings = EncList(encodings)

        writer.write(b'\x01')
        width = await read_int(reader, 2)
        height = await read_int(reader, 2)
        mode_data = bytearray(await reader.readexactly(13))
        mode_data[2] &= 1  # set big endian flag to 0 or 1
        mode_data[3] &= 1  # set true colour flag to 0 or 1
        mode = video_modes.get(bytes(mode_data))
        await reader.readexactly(3)  # padding
        name = await read_text(reader, 'utf-8')

        if mode is None:
            mode = 'rgba'
            writer.write(b'\x00\x00\x00\x00' + video_definition.get(mode) + b'\x00\x00\x00')

        writer.write(b'\x02\x00'+len(encodings).to_bytes(2, 'big'))
        # signed=True to allow pseudo-encodings (negative values per RFB spec).
        writer.write(b''.join(map(lambda x: x.value.to_bytes(4, 'big', signed=True), encodings)))

        decompress = decompressobj()

        return cls(reader, writer, decompress, name, width, height, mode, encodings)

    def get_rect(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> tuple:
        """
        Crops the rectangle according to the video buffer.
        """
        if x < 0:
            x = 0
        elif x > self.width:
            x = self.width
        if y < 0:
            y = 0
        elif y > self.height:
            y = self.height
        if (width is None) or (width + x > self.width):
            width = self.width - x
        elif width < 0:
            width = 0
        if (height is None)  or (height + y > self.height):
            height = self.height - y
        elif height < 0:
            height = 0

        return (x, y, width, height)

    def refresh(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        """
        Sends a video buffer update request to the server.
        """

        incremental = self.data is not None

        (x, y, width, height) = self.get_rect(x, y, width, height)

        self.writer.write(
            b'\x03' +
            incremental.to_bytes(1, 'big') +
            x.to_bytes(2, 'big') +
            y.to_bytes(2, 'big') +
            width.to_bytes(2, 'big') +
            height.to_bytes(2, 'big'))

    def _handle_desktop_resize(self, new_width: int, new_height: int):
        """
        Apply a DesktopSize / ExtendedDesktopSize pseudo-encoding rect.

        Only invalidates the framebuffer on an actual size change. Some
        servers echo the current size as an informational EDS rect even
        when nothing changed; treating that as "data invalid + need
        refresh" would deadlock the screenshot path.

        Does not auto-request a refresh -- callers (Client.screenshot)
        check whether their region became incomplete and re-request as
        needed. This keeps the rect handler free of side effects on the
        writer side, which would otherwise race with in-flight FBURs.
        """
        if (new_width, new_height) == (self.width, self.height):
            return
        self.width = new_width
        self.height = new_height
        self.data = None
        self.serial = (self.serial + 1) & 0xfffffff

    def _update_rect(self, x1: int, x2: int, y1: int, y2: int, data: np.ndarray):
        """
        Fills the space of the rectangle with the selected data
        Accepts various input shapes:
            HxWx4 -- raw copy
            1x1x4 -- fills with one color
            HxWx3 -- adds alpha
            1x1x3 -- fills with one color and adds alpha
        """

#        print(f"_update_rect {x1} +{x2-x1} {y1} +{y2-y1} {data.shape}")
        if self.data is None:
            self.data = np.zeros((self.height, self.width, 4), 'B')
        a_index = self.mode.index('a')
        if data.shape[2] == 4:
            self.data[y1:y2, x1:x2, :] = data
        if data.shape[2] == 3:
            if a_index:
                self.data[y1:y2, x1:x2, 0:3] = data
            else:
                self.data[y1:y2, x1:x2, 1:4] = data
        self.data[y1:y2, x1:x2, a_index] = 255
        self.serial = (self.serial + 1) & 0xfffffff

    async def process_raw(self, _reader, x, y, width, height):
        ff = height * width * 4
        (cx, cy) = (x, y)
        data = b''
        while ff:
            newdata = await _reader.read(ff)
            ff -= len(newdata)
            data += newdata
            rows = len(data) // (width * 4)
            if rows > 0:
                await sleep(0)
                self._update_rect(cx, cx + width, cy, cy + rows, np.ndarray((rows, width, 4), 'B', data))
                cy += rows
                data = data[rows * width * 4:]
                await sleep(0)

    async def process_xrle(self, _reader, x, y, width, height, tile_sz):
        palette = list()

        for (cx, cy, cw, ch) in _tile_gen(tile_sz, tile_sz, x, y, width, height):
            subencoding  = await read_int(_reader, 1)
            await sleep(0)
            if subencoding == 0:
                block = await _reader.readexactly(ch * cw * 3)
                self._update_rect(cx, cx + cw, cy, cy + ch, np.ndarray((ch, cw, 3), 'B', block))
            elif subencoding == 1:
                block = await _reader.readexactly(3)
                self._update_rect(cx, cx + cw, cy, cy + ch, np.ndarray((1, 1, 3), 'B', block))
            elif (subencoding <= 16) or (subencoding == 127):
                block = await _rle_packedbits(_reader, cw, ch, subencoding, palette)
                self._update_rect(cx, cx + cw, cy, cy + ch, np.ndarray((ch, cw, 3), 'B', block))
            elif subencoding < 128:
                raise ValueError(f"Palette {subencoding} is forbidden")
            else:
                block = await _rle_rle(_reader, cw, ch, subencoding, palette)
                self._update_rect(cx, cx + cw, cy, cy + ch, np.ndarray((ch, cw, 3), 'B', block))
            await sleep(0)

    async def process_copy(self, _reader, x, y, width, height):
        srcx = await read_int(_reader, 2)
        srcy = await read_int(_reader, 2)
#        print(f"Copy Rect {width}x{height} ({srcx},{srcy}) ==> ({x},{y})")
        self._update_rect(x, x + width, y, y + height, self.data[srcy:srcy + height, srcx:srcx + width, :])


    async def read(self):
        x = await read_int(self.reader, 2)
        y = await read_int(self.reader, 2)
        width = await read_int(self.reader, 2)
        height = await read_int(self.reader, 2)
        # Encoding number is signed 32-bit on the wire. read_int is unsigned,
        # so convert wrap-around values back to negative for pseudo-encodings.
        raw_enc = await read_int(self.reader, 4)
        if raw_enc > 0x7FFFFFFF:
            raw_enc -= 0x100000000
        encoding = Enc(raw_enc)
        length = height * width * 4
#        print(f"GET VIDEO REC: {x} {y} {width}x{height} / {encoding}")

        if encoding is Enc.RAW:
            await self.process_raw(self.reader,
                       x, y, width, height)

        elif encoding is Enc.ZLIB:
            length = await read_int(self.reader, 4)
            await self.process_raw(StreamZReader(self.reader, self.decompress, length),
                       x, y, width, height)

        elif encoding is Enc.TRLE:
            await self.process_xrle(self.reader,
                       x, y, width, height, 16)

        elif encoding is Enc.ZRLE:
            length = await read_int(self.reader, 4)
            await self.process_xrle(StreamZReader(self.reader, self.decompress, length),
                       x, y, width, height, 64)

        elif encoding is Enc.COPY:
            await self.process_copy(self.reader,
                                 x, y, width, height)

        elif encoding is Enc.DESKTOP_SIZE:
            # No payload -- rect.width/height carry the new dimensions.
            self._handle_desktop_resize(width, height)

        elif encoding is Enc.EXTENDED_DESKTOP_SIZE:
            # Payload: num_screens (u8) + 3 padding bytes, then per-screen
            # records (16 bytes each: id u32, x u16, y u16, w u16, h u16,
            # flags u32). We don't surface per-screen info; rect.width/height
            # carry the new framebuffer dimensions.
            num_screens = await read_int(self.reader, 1)
            await self.reader.readexactly(3)  # reserved padding
            await self.reader.readexactly(num_screens * 16)
            self._handle_desktop_resize(width, height)

        else:
            raise ValueError(encoding)

    def as_rgba(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """
        Returns the video buffer or the selected part of it as a 3D RGBA array.
        """

        (x, y, width, height) = self.get_rect(x, y, width, height)

        if self.data is None:
            return np.zeros((height, width, 4), 'B')
        if self.mode == 'rgba':
            return self.data[y:y + height, x:x + width, :]
        if self.mode == 'abgr':
            return self.data[y:y + height, x:x + width, ::-1]
        return np.dstack((
            self.data[y:y + height, x:x + width, self.mode.index('r')],
            self.data[y:y + height, x:x + width, self.mode.index('g')],
            self.data[y:y + height, x:x + width, self.mode.index('b')],
            self.data[y:y + height, x:x + width, self.mode.index('a')]))

    def is_complete(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        """
        Returns true if the video buffer or the selected part of it is entirely opaque.
        """

        if self.data is None:
            return False

        (x, y, width, height) = self.get_rect(x, y, width, height)

        return self.data[y:y + height, x:x + width, self.mode.index('a')].all()

    def detect_screens(self) -> List[Screen]:
        """
        Detect physical screens by inspecting the alpha channel.
        """

        if self.data is None:
            return []

        mask = self.data[:, :, self.mode.index('a')]
        mask = np.pad(mask // 255, ((1, 1), (1, 1))).astype(np.int8)
        mask_a = mask[1:, 1:]
        mask_b = mask[1:, :-1]
        mask_c = mask[:-1, 1:]
        mask_d = mask[:-1, :-1]

        screens = []
        while True:
            # Detect corners by ANDing perpendicular pairs of differences.
            corners = product(
                np.argwhere(mask_b - mask_a & mask_c - mask_a == -1),  # top left
                np.argwhere(mask_a - mask_b & mask_d - mask_b == -1),  # top right
                np.argwhere(mask_d - mask_c & mask_a - mask_c == -1),  # bottom left
                np.argwhere(mask_c - mask_d & mask_b - mask_d == -1))  # bottom right

            # Find cases where 3 corners align, forming an  'L' shape.
            rects = set()
            for a, b, c, d in corners:
                ab = a[0] == b[0] and a[1] < b[1]  # top
                cd = c[0] == d[0] and c[1] < d[1]  # bottom
                ac = a[1] == c[1] and a[0] < c[0]  # left
                bd = b[1] == d[1] and b[0] < d[0]  # right
                if ab and ac:
                    rects.add((a[1], a[0], b[1], c[0]))
                if ab and bd:
                    rects.add((a[1], a[0], d[1], d[0]))
                if cd and ac:
                    rects.add((a[1], a[0], d[1], d[0]))
                if cd and bd:
                    rects.add((c[1], b[0], d[1], d[0]))

            # Create screen objects and sort them by their scores.
            candidates = [Screen(int(x0), int(y0), int(x1 - x0), int(y1 - y0)) for x0, y0, x1, y1 in rects]
            candidates.sort(key=lambda screen: screen.score, reverse=True)

            # Find a single fully-opaque screen
            for screen in candidates:
                if mask_a[screen.slices].all():
                    mask_a[screen.slices] = 0
                    screens.append(screen)
                    break

            # Finish up if no screens remain
            else:
                return screens


class UpdateType(Enum):
    """
    Update from server to client.
    """

    #: Video update.
    VIDEO = 0

    #: Bell update.
    BELL = 2

    #: Clipboard update.
    CLIPBOARD = 3


@dataclass
class Client:
    """
    VNC client.
    """

    reader: StreamReader = field(repr=False)
    writer: StreamWriter = field(repr=False)

    #: The shared clipboard.
    clipboard: Clipboard

    #: The virtual keyboard.
    keyboard: Keyboard

    #: The virtual mouse.
    mouse: Mouse

    #: The video buffer.
    video: Video

    #: The server's public key (Mac only)
    host_key: Optional[rsa.RSAPublicKey]

    @classmethod
    async def create(
            cls,
            reader: StreamReader,
            writer: StreamWriter,
            username: Optional[str] = None,
            password: Optional[str] = None,
            host_key: Optional[rsa.RSAPublicKey] = None,
            encodings: Optional[list] = None) -> 'Client':


        intro = await reader.readline()
        if intro[:4] != b'RFB ':
            raise ValueError('not a VNC server')
        writer.write(b'RFB 003.008\n')

        auth_types = set(await reader.readexactly(await read_int(reader, 1)))
        if not auth_types:
            raise ValueError(await read_text(reader, 'utf-8'))
        for auth_type in (33, 1, 2):
            if auth_type in auth_types:
                writer.write(auth_type.to_bytes(1, 'big'))
                break
        else:
            raise ValueError(f'unsupported auth types: {auth_types}')

        # Apple authentication
        if auth_type == 33:
            if username is None or password is None:
                raise ValueError('server requires username and password')
            if host_key is None:
                writer.write(b'\x00\x00\x00\x0a\x01\x00RSA1\x00\x00\x00\x00')
                await reader.readexactly(4)  # packet length
                await reader.readexactly(2)  # packet version
                host_key_length = await read_int(reader, 4)
                host_key = await reader.readexactly(host_key_length)
                host_key = load_der_public_key(host_key)
                await reader.readexactly(1)  # unknown
            aes_key = urandom(16)
            cipher = Cipher(algorithms.AES(aes_key), modes.ECB())
            encryptor = cipher.encryptor()
            credentials = pack_ard(username) + pack_ard(password)
            writer.write(
                b'\x00\x00\x01\x8a\x01\x00RSA1' +
                b'\x00\x01' + encryptor.update(credentials) +
                b'\x00\x01' + host_key.encrypt(aes_key, padding=padding.PKCS1v15()))
            await reader.readexactly(4)  # unknown

        # VNC authentication
        if auth_type == 2:
            if password is None:
                raise ValueError('server requires password')
            des_key = password.encode('ascii')[:8].ljust(8, b'\x00')
            des_key = bytes(int(bin(n)[:1:-1].ljust(8, '0'), 2) for n in des_key)
            encryptor = Cipher(algorithms.TripleDES(des_key), modes.ECB()).encryptor()
            challenge = await reader.readexactly(16)
            writer.write(encryptor.update(challenge) + encryptor.finalize())

        auth_result = await read_int(reader, 4)
        if auth_result == 0:
            return cls(
                reader=reader,
                writer=writer,
                host_key=host_key,
                clipboard=Clipboard(writer),
                keyboard=Keyboard(writer),
                mouse=Mouse(writer),
                video=await Video.create(reader, writer, encodings))
        elif auth_result == 1:
            raise PermissionError('Auth failed')
        elif auth_result == 2:
            raise PermissionError('Auth failed (too many attempts)')
        else:
            reason = await reader.readexactly(auth_result)
            raise PermissionError(reason.decode('utf-8'))

    async def read(self) -> UpdateType:
        """
        Reads an update from the server and returns its type.
        """

        update_type = UpdateType(await read_int(self.reader, 1))

        if update_type is UpdateType.CLIPBOARD:
            await self.reader.readexactly(3)  # padding
            self.clipboard.text = await read_text(self.reader, 'latin-1')

        if update_type is UpdateType.VIDEO:
            await self.reader.readexactly(1)  # padding
            cnt = await read_int(self.reader, 2)
#            print(f"UpdateType.VIDEO count={cnt}")
            for _ in range(cnt):
                await self.video.read()

        return update_type

    async def drain(self):
        """
        Waits for data to be written to the server.
        """

        await self.writer.drain()

    async def screenshot(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        """
        Takes a screenshot and returns a 3D RGBA array.
        """

        self.video.data = None
        self.video.refresh(x, y, width, height)
        while True:
            update_type = await self.read()
            if update_type is UpdateType.VIDEO:
                if self.video.is_complete(x, y, width, height):
                    return self.video.as_rgba(x, y, width, height)
                # FBU arrived but didn't fill the requested region. Common
                # causes: an unsolicited DESKTOP_SIZE/EDS rect invalidated
                # the buffer mid-stream, or the server split the response.
                # Re-request to keep the loop making progress.
                self.video.refresh(x, y, width, height)


@asynccontextmanager
async def connect(
        host: str,
        port: int = 5900,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host_key: Optional[rsa.RSAPublicKey] = None,
        encodings: Optional[list] = None,
        opener=None):
    """
    Make a VNC client connection. This is an async context manager that returns a connected :class:`Client` instance.
    """

    opener = opener or open_connection
    reader, writer = await opener(host, port)

    client = await Client.create(reader, writer, username, password, host_key, encodings)
    try:
        yield client
    finally:
        writer.close()
        await writer.wait_closed()
