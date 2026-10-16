import struct
import zlib
import pytest
from image_ocr import png_from_rgb


def test_png_has_only_controller_generated_structure():
    pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    data = png_from_rgb(2, 2, pixels)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    chunks = []
    at = 8
    while at < len(data):
        size = struct.unpack(">I", data[at : at + 4])[0]
        kind = data[at + 4 : at + 8]
        payload = data[at + 8 : at + 8 + size]
        chunks.append((kind, payload))
        at += size + 12
    assert [k for k, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert zlib.decompress(chunks[1][1]) == b"\0" + pixels[:6] + b"\0" + pixels[6:]


@pytest.mark.parametrize(
    "w,h,data",
    [(True, 1, b"abc"), (1, 1, b"abcd"), (1, 1, b"ab"), (3000, 3000, b""), (0, 1, b"")],
)
def test_invalid_raw_pixel_protocol_is_rejected(w, h, data):
    with pytest.raises(ValueError):
        png_from_rgb(w, h, data)
