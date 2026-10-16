#!/usr/bin/env python3
"""
Produce allreducegemm/kernels.hsaco by binary-patching the upstream
amdappsdk/matrixmultiplication/kernels.hsaco.

The bug
-------
In the upstream MatrixMultiplication_Kernels.cl, mmmKernel_local computes the
row of A each work-item stages into local memory from the *local* Y id:

    int globalPosA = i * lSizeX + lIdX + (lIdY << TILEY_SHIFT) * temp;

so every workgroup row reads its A tile from the first 4*lSizeY rows of A,
while it writes C at its *global* row (gIdY). With more than one
workgroup row in the grid, every band but the first is wrong (e.g. 75% wrong
at M=128 with 8x8 workgroups: 4 bands, only band 0 correct). The fix is
gIdY in place of lIdY; see native/MatrixMultiplication_Kernels.cl.

Why a binary patch instead of recompiling
-----------------------------------------
MGPUSim v3 loads code object v2 (amd_kernel_code_t prefix), which current
LLVM no longer emits. Patching the shipped binary also keeps every other
instruction -- and therefore the timing behaviour -- identical to upstream,
so results stay comparable with earlier runs.

The patch
---------
Upstream, at 0x124c:
    v_lshlrev_b32 v5, 2, v1          ; v5 = lIdY << 2   (also used for blockPos)
    v_mul_lo_u32  v6, s16, v5        ; v6 = temp * (lIdY << 2)

Patched:
    v_lshlrev_b32 v5, 2, v1          ; unchanged, still feeds blockPos
    s_waitcnt     vmcnt(0)           ; v3 = lSizeY comes from a flat_load_ushort
    v_mad_u32_u24 v6, s9, v3, v1     ; wgIdY * lSizeY + lIdY
    v_add_u32     v6, vcc, s14, v6   ; + HiddenGlobalOffsetY      -> gIdY
    v_lshlrev_b32 v6, 2, v6          ; gIdY << TILEY_SHIFT
    v_mul_lo_u32  v6, s16, v6        ; v6 = temp * (gIdY << 2)

This is exactly how the kernel's own epilogue computes gIdY for globalPos
(s9 = workgroup id Y, v3 = workgroup_size_y, v1 = local id Y, s14 = low dword
of HiddenGlobalOffsetY); none of those registers is written before 0x1250.
Inserting 20 bytes shifts everything after 0x1250, so the one branch that
crosses the insertion point (s_cbranch_vccnz BB0_6 at 0x1230) is re-targeted,
and the .text section, PT_LOAD segment, kernel symbol size and BB labels are
updated. .text has 0x868 bytes of file padding after it, so no other section
moves.

Usage:  python3 patch_gidy.py            (paths default to this repo layout)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent.parent / "amdappsdk" / "matrixmultiplication" / "kernels.hsaco"
DEFAULT_DST = HERE.parent / "kernels.hsaco"

INSERT_AT = 0x1250  # address (== file offset) of v_mul_lo_u32 v6, s16, v5
BRANCH_AT = 0x1230  # s_cbranch_vccnz BB0_6, the only branch crossing INSERT_AT

INSERTED = [
    0xBF8C0F70,              # s_waitcnt vmcnt(0)
    0xD1C30006, 0x04060609,  # v_mad_u32_u24 v6, s9, v3, v1
    0x320C0C0E,              # v_add_u32_e32 v6, vcc, s14, v6
    0x240C0C82,              # v_lshlrev_b32_e32 v6, 2, v6
]
SHIFT = 4 * len(INSERTED)

# (address, expected upstream dword, patched dword), addresses pre-insertion.
REWRITES = [
    (0x124C, 0x240A0282, 0x240A0282),  # v_lshlrev_b32 v5, 2, v1 (sanity only)
    (0x1250, 0xD2850006, 0xD2850006),  # v_mul_lo_u32 v6, ... (word 0)
    (0x1254, 0x00020A10, 0x00020C10),  # src1 v5 -> v6
    (BRANCH_AT, 0xBF87011D, 0xBF87011D + len(INSERTED)),
]


def u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def patch(data: bytes) -> bytes:
    buf = bytearray(data)

    e_phoff, = struct.unpack_from("<Q", buf, 0x20)
    e_shoff, = struct.unpack_from("<Q", buf, 0x28)
    e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
        "<HHHHH", buf, 0x36)

    def shdr(i):
        base = e_shoff + i * e_shentsize
        name, typ, flags, addr, off, size, link, info, align, entsize = (
            struct.unpack_from("<IIQQQQIIQQ", buf, base))
        return base, name, typ, addr, off, size, entsize

    strtab_off = shdr(e_shstrndx)[4]

    def secname(i):
        n = shdr(i)[1]
        end = buf.index(b"\0", strtab_off + n)
        return buf[strtab_off + n:end].decode()

    secs = {secname(i): shdr(i) for i in range(e_shnum)}
    text_base, _, _, text_addr, text_off, text_size, _ = secs[".text"]
    if text_addr != text_off:
        raise SystemExit("unexpected layout: .text address != file offset")
    if text_size != 0x798:
        raise SystemExit(f"unexpected .text size {text_size:#x}; already patched?")

    for addr, want, _ in REWRITES:
        got = u32(buf, addr)
        if got != want:
            raise SystemExit(f"{addr:#x}: expected {want:#010x}, found {got:#010x}")

    text_end = text_off + text_size
    next_off = min(s[4] for n, s in secs.items() if s[4] > text_off and s[5])
    if text_end + SHIFT > next_off or any(buf[text_end:text_end + SHIFT]):
        raise SystemExit("no free padding after .text")

    for addr, _, new in REWRITES:
        struct.pack_into("<I", buf, addr, new)

    code = bytes(buf[INSERT_AT:text_end])
    ins = b"".join(struct.pack("<I", w) for w in INSERTED)
    buf[INSERT_AT:text_end + SHIFT] = ins + code

    # .text sh_size
    struct.pack_into("<Q", buf, text_base + 0x20, text_size + SHIFT)

    # PT_LOAD covering .text: p_filesz, p_memsz
    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        p_type, _, p_off = struct.unpack_from("<IIQ", buf, base)
        if p_type == 1 and p_off == text_off:
            filesz, memsz = struct.unpack_from("<QQ", buf, base + 0x20)
            struct.pack_into("<QQ", buf, base + 0x20, filesz + SHIFT, memsz + SHIFT)

    # Symbols: grow the kernel, move labels past the insertion point.
    for name in (".symtab", ".dynsym"):
        _, _, _, _, off, size, entsize = secs[name]
        for s in range(off, off + size, entsize):
            value, sz = struct.unpack_from("<QQ", buf, s + 8)
            if value == text_off and sz == text_size:
                struct.pack_into("<Q", buf, s + 16, sz + SHIFT)
            elif INSERT_AT <= value < text_end:
                struct.pack_into("<Q", buf, s + 8, value + SHIFT)

    return bytes(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = ap.parse_args()

    args.dst.write_bytes(patch(args.src.read_bytes()))
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
