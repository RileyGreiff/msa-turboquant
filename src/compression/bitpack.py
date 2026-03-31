"""Bit-packing utilities for sub-byte quantized tensors.

Packs unsigned integer values (each using `bits` bits) into a compact
uint8 byte stream, and unpacks them back. This makes compressed_bytes
reflect actual storage rather than the wasteful PyTorch dtype.

Supported bit widths:
- 1, 2, 4, 8: Fast path — values_per_byte = 8 // bits, shift+mask.
- 3, 5 (and any other): General bitstream path.
"""

from __future__ import annotations

import torch


def pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned integer values into a compact uint8 byte stream.

    Args:
        values: Tensor of unsigned ints in [0, 2^bits - 1], any shape/dtype.
        bits: Bits per value (1-8).

    Returns:
        1-D torch.uint8 tensor of packed bytes.
    """
    flat = values.flatten().to(torch.uint8)
    n = flat.numel()

    if bits == 8:
        return flat

    if 8 % bits == 0:
        return _pack_even(flat, bits, n)
    return _pack_general(flat, bits, n)


def unpack(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    """Unpack a uint8 byte stream back to individual unsigned integer values.

    Args:
        packed: 1-D uint8 tensor from pack().
        bits: Bits per value (must match what was used to pack).
        num_values: Original number of values (needed to trim padding).

    Returns:
        1-D torch.uint8 tensor of length num_values.
    """
    if bits == 8:
        return packed[:num_values]

    if 8 % bits == 0:
        return _unpack_even(packed, bits, num_values)
    return _unpack_general(packed, bits, num_values)


# ---------------------------------------------------------------------------
# Fast path: bit widths that evenly divide 8 (1, 2, 4)
# ---------------------------------------------------------------------------

def _pack_even(flat: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    vpb = 8 // bits  # values per byte
    # Pad to multiple of vpb
    pad = (-n) % vpb
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
    grouped = flat.reshape(-1, vpb)
    # Shift each value into position and OR together
    result = torch.zeros(grouped.shape[0], dtype=torch.uint8, device=flat.device)
    for i in range(vpb):
        shift = (vpb - 1 - i) * bits
        result |= grouped[:, i].to(torch.uint8) << shift
    return result


def _unpack_even(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    vpb = 8 // bits
    mask = (1 << bits) - 1
    # Extract each value position
    parts = []
    for i in range(vpb):
        shift = (vpb - 1 - i) * bits
        parts.append((packed >> shift) & mask)
    # Interleave: stack along dim=1 then flatten
    result = torch.stack(parts, dim=1).flatten().to(torch.uint8)
    return result[:num_values]


# ---------------------------------------------------------------------------
# General path: arbitrary bit widths (3, 5, etc.)
# ---------------------------------------------------------------------------

def _pack_general(flat: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    # Convert each value to its bit representation
    # bit_matrix shape: (n, bits) — MSB first
    shifts = torch.arange(bits - 1, -1, -1, device=flat.device)
    bit_matrix = ((flat.unsqueeze(1) >> shifts) & 1).to(torch.uint8)

    # Flatten to a 1-D bit stream
    bit_stream = bit_matrix.flatten()

    # Pad to multiple of 8
    total_bits = bit_stream.numel()
    pad = (-total_bits) % 8
    if pad > 0:
        bit_stream = torch.cat([bit_stream, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])

    # Reshape to groups of 8 and combine into bytes
    byte_groups = bit_stream.reshape(-1, 8)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=flat.device)
    packed = (byte_groups * weights).sum(dim=1).to(torch.uint8)
    return packed


def _unpack_general(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    # Expand each byte to 8 bits
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed.device)
    bit_stream = ((packed.unsqueeze(1) & weights) > 0).to(torch.uint8).flatten()

    # Take only the bits we need
    total_bits = num_values * bits
    bit_stream = bit_stream[:total_bits]

    # Reshape to (num_values, bits) and combine
    bit_matrix = bit_stream.reshape(num_values, bits)
    shifts = torch.arange(bits - 1, -1, -1, device=packed.device)
    values = (bit_matrix << shifts).sum(dim=1).to(torch.uint8)
    return values
