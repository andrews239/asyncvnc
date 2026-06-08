from asyncio import StreamReader
from io import BytesIO

import numpy as np
import pytest

from asyncvnc2 import Enc, Video


def _make_video(reader=None, width=1024, height=768) -> Video:
    """Build a Video for unit testing -- no live VNC connection."""
    return Video(
        reader=reader,
        writer=BytesIO(),
        decompress=lambda data: data,
        name='DESKTOP',
        width=width,
        height=height,
        mode='RGBA')


def _rect_header(x: int, y: int, width: int, height: int, encoding: int) -> bytes:
    """Encode a framebuffer-update rectangle header (12 bytes)."""
    return (
        x.to_bytes(2, 'big') +
        y.to_bytes(2, 'big') +
        width.to_bytes(2, 'big') +
        height.to_bytes(2, 'big') +
        encoding.to_bytes(4, 'big', signed=True))


def _reader_with(data: bytes) -> StreamReader:
    """StreamReader pre-loaded with canned bytes. Requires a running loop."""
    reader = StreamReader()
    reader._buffer.extend(bytearray(data))
    return reader


# -- _handle_desktop_resize -----------------------------------------------

def test_handle_desktop_resize_noop_on_unchanged_size():
    video = _make_video()
    video.data = np.zeros((video.height, video.width, 4), 'B')
    serial_before = video.serial
    video._handle_desktop_resize(video.width, video.height)
    assert video.data is not None  # not invalidated
    assert video.serial == serial_before  # not bumped


def test_handle_desktop_resize_invalidates_on_change():
    video = _make_video()
    video.data = np.zeros((video.height, video.width, 4), 'B')
    serial_before = video.serial
    video._handle_desktop_resize(1280, 720)
    assert video.width == 1280
    assert video.height == 720
    assert video.data is None  # framebuffer invalidated
    assert video.serial == serial_before + 1


# -- Video.read with DESKTOP_SIZE pseudo-encoding -------------------------

@pytest.mark.asyncio
async def test_read_desktop_size_rect():
    rect = _rect_header(0, 0, 800, 600, Enc.DESKTOP_SIZE.value)
    video = _make_video(reader=_reader_with(rect))
    await video.read()
    assert (video.width, video.height) == (800, 600)
    assert video.data is None


# -- Video.read with EXTENDED_DESKTOP_SIZE pseudo-encoding ----------------

@pytest.mark.asyncio
async def test_read_extended_desktop_size_drains_payload():
    # EDS rect: header + (num_screens=1) + 3 padding bytes + one 16-byte
    # screen record. After read() returns, the buffer must be empty so the
    # next rect's parser doesn't see leftover bytes.
    screen_record = (
        b'\x00\x00\x00\x01' +  # screen id
        b'\x00\x00' + b'\x00\x00' +  # x, y position
        (1600).to_bytes(2, 'big') +  # width
        (900).to_bytes(2, 'big') +   # height
        b'\x00\x00\x00\x00')         # flags
    payload = b'\x01' + b'\x00\x00\x00' + screen_record
    rect = _rect_header(0, 0, 1600, 900, Enc.EXTENDED_DESKTOP_SIZE.value) + payload
    video = _make_video(reader=_reader_with(rect))
    await video.read()
    assert (video.width, video.height) == (1600, 900)
    assert len(video.reader._buffer) == 0  # full payload consumed


@pytest.mark.asyncio
async def test_read_extended_desktop_size_multiple_screens():
    # Two screens -> payload size is 1 + 3 + 32 = 36 bytes after header.
    screen_record = b'\x00' * 16
    payload = b'\x02' + b'\x00\x00\x00' + (screen_record * 2)
    rect = _rect_header(0, 0, 2560, 1080, Enc.EXTENDED_DESKTOP_SIZE.value) + payload
    video = _make_video(reader=_reader_with(rect))
    await video.read()
    assert (video.width, video.height) == (2560, 1080)
    assert len(video.reader._buffer) == 0
