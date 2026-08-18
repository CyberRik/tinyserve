/* Differential replay driver.
 *
 * Reads an operation script on stdin and prints one result line per op, so the
 * C++ tree can be compared against tinyserve/prefix/cache.py's fallback without
 * loading it into the interpreter.
 *
 * This exists because "the Python fallback behaves identically" is a claim the
 * ctypes differential fuzz can only check where a *loadable* shared library
 * exists -- which excludes any host whose C++ toolchain does not match the
 * interpreter's ABI (a 32-bit MinGW against 64-bit CPython, say). Piping a
 * script through a subprocess has no ABI to match, so the two implementations
 * can still be held to the same trace anywhere a compiler exists at all.
 *
 * Script format, one op per line:
 *   block <n>            reset, with block size n
 *   insert <seq> <t>...  -> "insert <published>"
 *   match <t>...         -> "match <n_tokens> <donor>"
 *   reuse <t>...         -> "match_for_reuse" equivalent; the clamp lives in
 *                           Python, so this reports the raw match and the
 *                           comparison script applies the clamp to both sides
 *   evict <seq>          -> "evict"
 *   stat                 -> "stat <nodes> <seqs>"
 */

#include "tinyserve/prefix_cache.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  ts_prefix_cache *cache = ts_prefix_cache_create(16);
  std::string line;

  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string op;
    if (!(in >> op)) {
      continue;
    }

    if (op == "block") {
      int32_t bs = 16;
      in >> bs;
      ts_prefix_cache_destroy(cache);
      cache = ts_prefix_cache_create(bs);
      std::printf("block %d\n", (int)bs);
    } else if (op == "insert") {
      int32_t seq = 0;
      in >> seq;
      std::vector<int32_t> tokens;
      int32_t t;
      while (in >> t) {
        tokens.push_back(t);
      }
      int32_t published = ts_prefix_cache_insert(
          cache, tokens.empty() ? NULL : &tokens[0], (int32_t)tokens.size(), seq);
      std::printf("insert %d\n", (int)published);
    } else if (op == "match" || op == "reuse") {
      std::vector<int32_t> tokens;
      int32_t t;
      while (in >> t) {
        tokens.push_back(t);
      }
      int32_t donor = TS_PREFIX_NO_SEQ;
      int32_t n = ts_prefix_cache_match(cache, tokens.empty() ? NULL : &tokens[0],
                                        (int32_t)tokens.size(), &donor);
      std::printf("match %d %d\n", (int)n, (int)donor);
    } else if (op == "evict") {
      int32_t seq = 0;
      in >> seq;
      ts_prefix_cache_evict(cache, seq);
      std::printf("evict\n");
    } else if (op == "stat") {
      std::printf("stat %d %d\n", (int)ts_prefix_cache_node_count(cache),
                  (int)ts_prefix_cache_seq_count(cache));
    }
    std::fflush(stdout);
  }

  ts_prefix_cache_destroy(cache);
  return 0;
}
