"""Bounded, versioned TCP framing. One frame in flight; receiver replies ACK."""
import json
import struct

HEADER = struct.Struct('!4sII')
MAGIC = b'HND1'
MAX_META = 16384
MAX_JPEG = 262144


def read_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        block = sock.recv(count - len(data))
        if not block:
            raise ConnectionError('Connection closed during frame')
        data.extend(block)
    return bytes(data)


def pack_frame(metadata, jpeg):
    raw = json.dumps(metadata, allow_nan=False, separators=(',', ':')).encode()
    if not 0 < len(raw) <= MAX_META or not 0 < len(jpeg) <= MAX_JPEG:
        raise ValueError('Frame too large or empty')
    return HEADER.pack(MAGIC, len(raw), len(jpeg)) + raw + jpeg


def receive_frame(sock):
    magic, meta_size, jpeg_size = HEADER.unpack(read_exact(sock, HEADER.size))
    if magic != MAGIC or not 0 < meta_size <= MAX_META or not 0 < jpeg_size <= MAX_JPEG:
        raise ValueError('Invalid frame header')
    meta = json.loads(read_exact(sock, meta_size))
    if not isinstance(meta, dict):
        raise ValueError('Metadata must be an object')
    return meta, read_exact(sock, jpeg_size)
