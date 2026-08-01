"""Block-based logical KV-cache accounting (PRD Sections 6.5 and 8).

This tracks *logical* ownership of llama.cpp's own (unified) KV-cache token
budget; it is an accounting layer, not a physical allocator — the actual
cells live inside the llama_context the Runtime creates with kv_unified=True,
sized to `total_tokens`. block_size is measured in tokens, matching
llama.cpp's own position-addressed cells.
"""

import math


class KVCacheManager:
    def __init__(self, total_tokens: int, block_size: int = 16) -> None:
        self._block_size = block_size
        self._total_blocks = total_tokens // block_size
        self._free_blocks = self._total_blocks
        self._block_table: dict[str, int] = {}

    def _blocks_needed(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self._block_size)

    def reserve(self, seq_id: str, n_tokens: int) -> bool:
        if seq_id in self._block_table:
            raise ValueError(f"seq_id {seq_id!r} already holds a reservation")
        needed = self._blocks_needed(n_tokens)
        if needed > self._free_blocks:
            return False
        self._free_blocks -= needed
        self._block_table[seq_id] = needed
        return True

    def release(self, seq_id: str) -> None:
        blocks = self._block_table.pop(seq_id, None)
        if blocks is not None:
            self._free_blocks += blocks

    def available_blocks(self) -> int:
        return self._free_blocks

    def total_blocks(self) -> int:
        return self._total_blocks
