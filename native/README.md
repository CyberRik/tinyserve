# `native/` — prefix cache in C++

A block-aligned radix tree over token IDs, answering one question:

> How much of this prompt has some other sequence already prefilled, and who
> holds those KV cells?

TinyServe uses the answer to call `llama_memory_seq_cp` and start a sequence at
a non-zero `n_past`, so the shared part of a prompt never reaches
`llama_decode()` a second time.

```
include/tinyserve/prefix_cache.h   the C ABI, and the rationale
src/prefix_cache.cpp               the tree
tests/test_prefix_cache.cpp        standalone correctness harness
tests/replay_driver.cpp            stdin/stdout driver for the differential test
CMakeLists.txt
```

## Why this is C++ and not Python

Not for CPU time, and the repo should not be read as claiming otherwise.
`docs/profiling-notes.md` measured **97.2%** of wall time inside
`llama_decode()`, and a direct benchmark of the ctypes batch-fill path put it at
**0.63 µs/row** — 0.64 ms even at a full 1024-row tick, under 2% of a 33.7 ms
decode. Rewriting TinyServe's Python bookkeeping in C++ would buy nothing
measurable, and that was checked before this directory existed rather than
assumed afterwards.

The reason it is C++ is that prefix reuse is the wrong thing to own in Python at
all. It belongs next to the KV cache, and llama.cpp's own server
(`tools/server/server.cpp`) is C++ — it keeps a `slot.cache_tokens` vector per
slot and does a linear common-prefix scan against it, which is O(slots ×
prompt) per request and finds nothing across slots. A block-aligned radix tree
is the standard fix; it is what vLLM's automatic prefix caching and SGLang's
RadixAttention do.

So the constraint this directory is built under: **no Python, no llama.cpp, no
dependency beyond the C++ standard library.** The same translation unit that
TinyServe loads over ctypes today links straight into a C++ server. A pybind11
extension would have been less code here and unusable there.

The gains are in work removed, not cycles shaved. Measured on the benchmark
below: **47–70% of prompt tokens never reached `llama_decode()`**.

## Building

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
ctest --test-dir native/build --output-on-failure
```

That produces `tinyserve_prefix.{so,dll,dylib}`, which
`tinyserve/prefix/cache.py` finds automatically. Point `TINYSERVE_PREFIX_LIB` at
it to override the search.

Nothing here is required to run TinyServe. `tinyserve/prefix/cache.py` ships a
pure-Python implementation with identical semantics and falls back to it when
the library is missing, stale, or built against a different ABI major version.
Which backend is live shows up as `prefix_cache_backend_info{backend=...}` on
`/metrics`, so a deployment that quietly fell back is visible on the dashboard
rather than a mystery in a latency graph.

## Testing

Three layers, because each catches something the others cannot:

| | what it covers |
|---|---|
| `native/tests/test_prefix_cache.cpp` | 37 assertions. NULL arguments and a 200k-deep teardown — the things ctypes cannot reach from Python. |
| `tests/unit/test_prefix_cache.py` | Behaviour, parametrised over *both* backends, plus a ctypes differential fuzz. Requires an ABI-compatible build. |
| `tests/unit/test_prefix_cache_differential.py` | Compiles `replay_driver.cpp` and pipes 4000 random ops through it, comparing the trace to the Python cache op-for-op. **No ABI match required** — it works on any host with a compiler, including a 32-bit toolchain against a 64-bit interpreter. |

The middle layer skips when the shared library is absent; the third still runs.
That matters: the fallback's correctness is load-bearing (`n_past` comes from
its answer), so "no toolchain" must mean "no speedup", never "different
behaviour", and that claim needs a test that survives a toolchain mismatch.

Correctness of the *reuse*, as opposed to the bookkeeping, is pinned separately
in `tests/integration/test_prefix_reuse.py`: a sequence that skips prefill by
copying KV cells must emit token-for-token what it would have emitted by
prefilling the prompt itself. A tree that is perfectly self-consistent can still
hand the batch loop an `n_past` pointing at cells holding something else, and
that failure mode is not a crash — it is fluent, plausible, wrong output.

## Thread safety

None. TinyServe drives this from the single-threaded batch loop. Callers that
need concurrency serialize externally; a mutex here would tax the
single-threaded case for nobody's benefit.
