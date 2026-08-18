/* TinyServe prefix cache -- block-aligned radix tree over token IDs.
 *
 * WHY THIS IS C++ AND NOT PYTHON
 * ------------------------------
 * Not for CPU time. docs/profiling-notes.md measured 97.2% of wall time inside
 * llama_decode(), and a direct benchmark of the ctypes batch-fill path put it at
 * ~0.63 us/row (0.64 ms at a full 1024-row tick, under 2% of a 33.7 ms decode).
 * Rewriting TinyServe's Python bookkeeping in C++ would buy approximately
 * nothing, and this file does not claim otherwise.
 *
 * The reason this is C++ is that it is the wrong layer to own in Python at all.
 * Prefix reuse belongs next to the KV cache, and llama.cpp's own server
 * (tools/server/server.cpp) is C++ -- it keeps a `slot.cache_tokens` vector and
 * does a linear common-prefix scan per slot, which is O(slots x prompt) per
 * request and finds nothing across slots that are not the one being reused.
 * A block-aligned radix tree over token IDs is the standard fix (vLLM's
 * automatic prefix caching, SGLang's RadixAttention). Written behind a flat C
 * ABI with no Python and no llama.cpp in its dependency set, the same
 * translation unit serves TinyServe over ctypes today and can be dropped into a
 * C++ server directly.
 *
 * That constraint is load-bearing, so it is worth stating plainly: this library
 * knows nothing about tokens beyond "int32 that compares equal", nothing about
 * KV cells, and nothing about who calls it. It answers one question -- "how much
 * of this prompt has someone already prefilled, and who holds it" -- and the
 * caller decides what to do with the answer.
 *
 * BLOCK ALIGNMENT
 * ---------------
 * Matches are reported in whole blocks of `block_size` tokens, never in partial
 * blocks. This is not an approximation for convenience: llama.cpp's KV reuse
 * primitive (llama_memory_seq_cp) copies a position range, and TinyServe's
 * KVCacheManager accounts in blocks of kv_block_size. Reporting a match that
 * does not land on a block boundary would hand the caller a number it cannot
 * act on. Keep block_size equal to TINYSERVE_KV_BLOCK_SIZE.
 *
 * THREAD SAFETY
 * -------------
 * None. TinyServe drives this from the single-threaded batch loop. Callers that
 * need concurrency must serialize externally -- adding a mutex here would tax
 * the single-threaded case for nobody's benefit.
 */

#ifndef TINYSERVE_PREFIX_CACHE_H
#define TINYSERVE_PREFIX_CACHE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(TINYSERVE_BUILD_SHARED)
#    define TINYSERVE_API __declspec(dllexport)
#  else
#    define TINYSERVE_API
#  endif
#else
#  if defined(TINYSERVE_BUILD_SHARED)
#    define TINYSERVE_API __attribute__((visibility("default")))
#  else
#    define TINYSERVE_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle. */
typedef struct ts_prefix_cache ts_prefix_cache;

/* Sentinel written to *out_seq_id when a match reports zero tokens. */
#define TS_PREFIX_NO_SEQ (-1)

/* Create a cache keyed in blocks of `block_size` tokens.
 * Returns NULL if block_size < 1. */
TINYSERVE_API ts_prefix_cache *ts_prefix_cache_create(int32_t block_size);

/* Destroy a cache. NULL is a no-op. */
TINYSERVE_API void ts_prefix_cache_destroy(ts_prefix_cache *cache);

/* Longest block-aligned prefix of `tokens` already resident.
 *
 * Returns the number of matched tokens (always a multiple of block_size) and
 * writes the donor sequence id to *out_seq_id, or TS_PREFIX_NO_SEQ when the
 * match is empty. `out_seq_id` may be NULL.
 *
 * The donor is the most recently inserted live owner of the deepest matched
 * block, which is the sequence most likely to still hold those KV cells.
 *
 * The final block of a prompt is deliberately reachable: a caller that matches
 * its entire prompt would have nothing left to prefill and no row to request
 * logits on. Callers must clamp -- TinyServe holds back one block, see
 * tinyserve/prefix/cache.py. Clamping here would bake a caller's constraint
 * into a general-purpose index. */
TINYSERVE_API int32_t ts_prefix_cache_match(ts_prefix_cache *cache,
                                            const int32_t *tokens,
                                            int32_t n_tokens,
                                            int32_t *out_seq_id);

/* Publish `tokens` as resident under `seq_id`, truncated down to a block
 * boundary. Returns the number of tokens actually published.
 *
 * Re-inserting the same seq_id replaces its previous claim (a sequence grows as
 * it decodes; the newer, longer path supersedes the older one).
 *
 * Publishing is the caller's decision, not this library's: TinyServe publishes
 * a prompt at admission, when its prefill is scheduled and its KV cells are
 * committed -- not when it completes. */
TINYSERVE_API int32_t ts_prefix_cache_insert(ts_prefix_cache *cache,
                                             const int32_t *tokens,
                                             int32_t n_tokens,
                                             int32_t seq_id);

/* Drop `seq_id`'s claim. Nodes are freed only once no owner and no surviving
 * descendant remains, so evicting a sequence never invalidates a prefix another
 * live sequence is still sharing. Unknown seq_id is a no-op. */
TINYSERVE_API void ts_prefix_cache_evict(ts_prefix_cache *cache, int32_t seq_id);

/* Live block-nodes, excluding the root. For tests and the
 * tinyserve_prefix_cache_nodes gauge -- this is what proves eviction actually
 * reclaims rather than leaking. */
TINYSERVE_API int32_t ts_prefix_cache_node_count(const ts_prefix_cache *cache);

/* Sequences holding at least one block. */
TINYSERVE_API int32_t ts_prefix_cache_seq_count(const ts_prefix_cache *cache);

/* ABI version, "major.minor". Checked by the ctypes loader so a stale .so on
 * the library path degrades to the Python fallback instead of crashing. */
TINYSERVE_API const char *ts_prefix_cache_abi_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TINYSERVE_PREFIX_CACHE_H */
