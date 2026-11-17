"""CRC32 for unitree_go/msg/LowCmd.

The robot rejects a LowCmd whose CRC does not match, so a mistake here means the
command is ignored -- the failure is silent but safe, never a runaway.

The checksum is Unitree's own bitwise CRC-32/MPEG-2 variant run over the packed
C struct as 32-bit words, excluding the trailing crc field. The struct layout is
reproduced below from unitree_go's message definition plus C alignment rules; the
module refuses to run if the packed size is not the expected 812 bytes.
"""

from __future__ import annotations

import struct

POLYNOMIAL = 0x04C11DB7
LOWCMD_BYTES = 812  # 203 uint32 words, the last of which is the crc itself


def crc32_core(words) -> int:
    crc = 0xFFFFFFFF
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ POLYNOMIAL
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= POLYNOMIAL
            xbit >>= 1
    return crc & 0xFFFFFFFF


def pack_lowcmd(cmd) -> bytes:
    """Serialise a LowCmd ROS2 message into the SDK's C struct layout."""
    buf = bytearray()
    buf += struct.pack("<2B", *(int(v) & 0xFF for v in cmd.head))
    buf += struct.pack("<BB", int(cmd.level_flag) & 0xFF, int(cmd.frame_reserve) & 0xFF)
    buf += struct.pack("<2I", *(int(v) & 0xFFFFFFFF for v in cmd.sn))
    buf += struct.pack("<2I", *(int(v) & 0xFFFFFFFF for v in cmd.version))
    buf += struct.pack("<H2x", int(cmd.bandwidth) & 0xFFFF)  # 2 bytes pad to align

    for m in cmd.motor_cmd:  # 20 entries, 36 bytes each
        buf += struct.pack("<B3x", int(m.mode) & 0xFF)
        buf += struct.pack("<5f", float(m.q), float(m.dq), float(m.tau), float(m.kp), float(m.kd))
        buf += struct.pack("<3I", *(int(v) & 0xFFFFFFFF for v in m.reserve))

    buf += struct.pack("<B3B", int(cmd.bms_cmd.off) & 0xFF,
                       *(int(v) & 0xFF for v in cmd.bms_cmd.reserve))
    buf += bytes(int(v) & 0xFF for v in cmd.wireless_remote)
    buf += bytes(int(v) & 0xFF for v in cmd.led)
    buf += bytes(int(v) & 0xFF for v in cmd.fan)
    # gpio sits at offset 802; a single pad byte aligns `reserve` to 804
    buf += struct.pack("<Bx", int(cmd.gpio) & 0xFF)
    buf += struct.pack("<I", int(cmd.reserve) & 0xFFFFFFFF)
    buf += struct.pack("<I", 0)  # crc placeholder

    if len(buf) != LOWCMD_BYTES:
        raise RuntimeError(
            f"packed LowCmd is {len(buf)} bytes, expected {LOWCMD_BYTES}. "
            "The struct layout does not match this firmware; refusing to send."
        )
    return bytes(buf)


def lowcmd_crc(cmd) -> int:
    data = pack_lowcmd(cmd)
    words = struct.unpack(f"<{LOWCMD_BYTES // 4}I", data)
    return crc32_core(words[:-1])
