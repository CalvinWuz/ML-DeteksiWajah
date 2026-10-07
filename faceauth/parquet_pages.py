"""Membaca gambar tertentu dari file parquet jarak jauh tanpa mengunduh seluruh file.

File OpenFake berukuran ~5 GB dan menyimpan banyak generator bercampur. Kolom gambarnya
tidak terkompresi dan dipecah menjadi halaman berisi 1-5 gambar, sehingga kita bisa:
  1. membaca header halaman (puluhan byte) untuk mengetahui jumlah baris tiap halaman,
  2. mengunduh hanya halaman yang berisi baris yang kita butuhkan.
"""
import struct
import time

import requests


# ---------------------------------------------------------------- Thrift compact protocol (minimal)
def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return r, i


def _zz(n):
    return (n >> 1) ^ -(n & 1)


def _value(b, i, t):
    if t in (1, 2):
        return t == 1, i
    if t == 3:
        return b[i], i + 1
    if t in (4, 5, 6):
        v, i = _varint(b, i)
        return _zz(v), i
    if t == 7:
        return b[i:i + 8], i + 8
    if t == 8:
        n, i = _varint(b, i)
        return b[i:i + n], i + n
    if t in (9, 10):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n, i = _varint(b, i)
        out = []
        for _ in range(n):
            if et in (1, 2):
                out.append(b[i] == 1)
                i += 1
            else:
                v, i = _value(b, i, et)
                out.append(v)
        return out, i
    if t == 12:
        return read_struct(b, i)
    raise ValueError(f"tipe thrift {t} tidak didukung")


def read_struct(b, i=0):
    fields, last = {}, 0
    while True:
        h = b[i]
        i += 1
        if h == 0:
            return fields, i
        delta, t = h >> 4, h & 0x0F
        if delta:
            fid = last + delta
        else:
            v, i = _varint(b, i)
            fid = _zz(v)
        last = fid
        fields[fid], i = _value(b, i, t)


# ---------------------------------------------------------------- HTTP range
class RemoteFile:
    """Range request ke URL Hugging Face; URL CDN hasil redirect di-cache dan diperbarui jika kedaluwarsa."""

    def __init__(self, url: str):
        self.url = url
        self.s = requests.Session()
        self._cdn = None

    def _resolve(self):
        r = self.s.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self._cdn = r.url

    def read(self, start: int, length: int, tries=5) -> bytes:
        for k in range(tries):
            try:
                if self._cdn is None:
                    self._resolve()
                r = self.s.get(self._cdn, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=120)
                if r.status_code in (401, 403):
                    self._cdn = None
                    raise requests.HTTPError("URL kedaluwarsa")
                r.raise_for_status()
                return r.content
            except (requests.RequestException, OSError):
                if k == tries - 1:
                    raise
                time.sleep(2 * (k + 1))


# ---------------------------------------------------------------- dekode halaman BYTE_ARRAY PLAIN
def _rle_levels(buf, bit_width, n):
    """Dekode RLE/bit-packed hybrid (definition levels)."""
    out, i = [], 0
    byte_w = (bit_width + 7) // 8
    while len(out) < n and i < len(buf):
        h, i = _varint(buf, i)
        if h & 1:  # bit-packed
            groups = h >> 1
            nbytes = groups * bit_width
            bits = int.from_bytes(buf[i:i + nbytes], "little")
            i += nbytes
            for k in range(groups * 8):
                out.append((bits >> (k * bit_width)) & ((1 << bit_width) - 1))
        else:  # run
            cnt = h >> 1
            v = int.from_bytes(buf[i:i + byte_w], "little")
            i += byte_w
            out.extend([v] * cnt)
    return out[:n]


def decode_byte_array_page(header: dict, body: bytes, max_def: int) -> list:
    """Nilai bytes per baris (None untuk null) dari satu halaman data tak terkompresi."""
    ptype = header[1]
    if ptype == 0:  # DataPage V1
        n = header[5][1]
        i = 0
        if max_def > 0:
            (ln,) = struct.unpack_from("<I", body, 0)
            levels = _rle_levels(body[4:4 + ln], max(1, max_def.bit_length()), n)
            i = 4 + ln
        else:
            levels = [0] * n
    elif ptype == 3:  # DataPage V2
        h2 = header[8]
        n, dl, rl = h2[1], h2[5], h2[6]
        lv = body[rl:rl + dl]
        levels = _rle_levels(lv, max(1, max_def.bit_length()), n) if max_def > 0 else [0] * n
        i = rl + dl
    else:
        raise ValueError(f"tipe halaman {ptype} tidak didukung")
    out = []
    for lev in levels:
        if lev < max_def:
            out.append(None)
            continue
        (ln,) = struct.unpack_from("<I", body, i)
        out.append(body[i + 4:i + 4 + ln])
        i += 4 + ln
    if i != len(body):
        raise ValueError(f"dekode halaman tidak pas: {i} != {len(body)}")
    return out


def iter_pages(rf: RemoteFile, start: int, total: int):
    """Header setiap halaman dalam satu column chunk: (offset_body, header_dict, num_values)."""
    off, end = start, start + total
    while off < end:
        b = rf.read(off, 256)
        try:
            h, hl = read_struct(b, 0)
        except IndexError:  # header lebih panjang (mis. berisi statistik): baca lebih banyak
            b = rf.read(off, 65536)
            h, hl = read_struct(b, 0)
        n = (h.get(5) or h.get(8) or {}).get(1, 0)
        yield off + hl, h, n
        off += hl + h[3]


def fetch_rows(rf: RemoteFile, chunk_start: int, chunk_size: int, wanted: set, max_def: int) -> dict:
    """{indeks baris dalam row group: bytes} hanya untuk baris di `wanted`."""
    got, row = {}, 0
    for body_off, h, n in iter_pages(rf, chunk_start, chunk_size):
        if h[1] == 2:  # halaman dictionary (tidak terpakai untuk PLAIN)
            continue
        rows = range(row, row + n)
        if wanted.intersection(rows):
            vals = decode_byte_array_page(h, rf.read(body_off, h[3]), max_def)
            for r, v in zip(rows, vals):
                if r in wanted:
                    got[r] = v
        row += n
        if len(got) == len(wanted):
            break
    return got
