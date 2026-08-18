/* Standalone correctness harness for the prefix cache.
 *
 * Deliberately links nothing but the C++ standard library -- no Python, no
 * llama.cpp, no test framework. That is the point of the flat C ABI: the same
 * translation unit that TinyServe loads over ctypes can be compiled and
 * exercised anywhere a C++ compiler exists, including toolchains that cannot
 * produce a loadable Python extension for the host interpreter.
 *
 *   g++ -std=c++11 -I../include ../src/prefix_cache.cpp test_prefix_cache.cpp -o test_prefix_cache
 *
 * The Python-side differential test (tests/unit/test_prefix_cache.py) asserts
 * the pure-Python fallback agrees with this implementation; these cases pin the
 * behaviour both must satisfy.
 */

#include "tinyserve/prefix_cache.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

static void check(bool ok, const char *what, long got, long want) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL  %s (got %ld, want %ld)\n", what, got, want);
  }
}

static void check_eq(long got, long want, const char *what) {
  check(got == want, what, got, want);
}

/* Distinct, deterministic token ids: block b of a stream is [seed+b*100 .. +N). */
static std::vector<int32_t> make_tokens(int32_t seed, int32_t n) {
  std::vector<int32_t> out;
  out.reserve((size_t)n);
  for (int32_t i = 0; i < n; ++i) {
    out.push_back(seed + i);
  }
  return out;
}

static int32_t match(ts_prefix_cache *c, const std::vector<int32_t> &t, int32_t *donor) {
  return ts_prefix_cache_match(c, t.empty() ? NULL : &t[0], (int32_t)t.size(), donor);
}

static int32_t insert(ts_prefix_cache *c, const std::vector<int32_t> &t, int32_t seq) {
  return ts_prefix_cache_insert(c, t.empty() ? NULL : &t[0], (int32_t)t.size(), seq);
}

/* --------------------------------------------------------------------- */

static void test_empty_cache_never_matches() {
  std::printf("test_empty_cache_never_matches\n");
  ts_prefix_cache *c = ts_prefix_cache_create(16);
  std::vector<int32_t> p = make_tokens(1000, 64);
  int32_t donor = 12345;
  check_eq(match(c, p, &donor), 0, "match on empty cache");
  check_eq(donor, TS_PREFIX_NO_SEQ, "donor reset to sentinel");
  check_eq(ts_prefix_cache_node_count(c), 0, "no nodes");
  ts_prefix_cache_destroy(c);
}

static void test_exact_and_partial_overlap() {
  std::printf("test_exact_and_partial_overlap\n");
  ts_prefix_cache *c = ts_prefix_cache_create(16);
  std::vector<int32_t> a = make_tokens(1000, 64); /* 4 blocks */
  check_eq(insert(c, a, 7), 64, "insert publishes 4 whole blocks");
  check_eq(ts_prefix_cache_node_count(c), 4, "4 nodes created");

  int32_t donor = TS_PREFIX_NO_SEQ;
  check_eq(match(c, a, &donor), 64, "identical prompt matches in full");
  check_eq(donor, 7, "donor is the inserting sequence");

  /* Shares the first 2 blocks (32 tokens), then diverges. */
  std::vector<int32_t> b = make_tokens(1000, 32);
  std::vector<int32_t> tail = make_tokens(9000, 32);
  b.insert(b.end(), tail.begin(), tail.end());
  donor = TS_PREFIX_NO_SEQ;
  check_eq(match(c, b, &donor), 32, "divergent prompt matches shared 2 blocks");
  check_eq(donor, 7, "donor named for the partial match");

  /* Diverges inside the first block -> nothing is block-aligned. */
  std::vector<int32_t> d = make_tokens(4242, 64);
  check_eq(match(c, d, NULL), 0, "no shared first block means no match");
  ts_prefix_cache_destroy(c);
}

static void test_matches_are_block_aligned() {
  std::printf("test_matches_are_block_aligned\n");
  ts_prefix_cache *c = ts_prefix_cache_create(16);

  /* 40 tokens = 2 whole blocks + 8 trailing; the tail is not publishable. */
  std::vector<int32_t> a = make_tokens(1000, 40);
  check_eq(insert(c, a, 1), 32, "insert truncates down to a block boundary");
  check_eq(match(c, a, NULL), 32, "match never reports the partial tail");

  /* Shorter than one block: nothing to say. */
  std::vector<int32_t> tiny = make_tokens(1000, 15);
  check_eq(insert(c, tiny, 2), 0, "sub-block insert publishes nothing");
  check_eq(match(c, tiny, NULL), 0, "sub-block match is empty");
  check_eq(ts_prefix_cache_seq_count(c), 1, "a seq owning nothing is not registered");
  ts_prefix_cache_destroy(c);
}

static void test_shared_prefix_survives_one_eviction() {
  std::printf("test_shared_prefix_survives_one_eviction\n");
  ts_prefix_cache *c = ts_prefix_cache_create(16);

  /* Two sequences share a 32-token system prompt, then diverge. */
  std::vector<int32_t> shared = make_tokens(500, 32);
  std::vector<int32_t> a = shared, b = shared;
  std::vector<int32_t> ta = make_tokens(7000, 32);
  std::vector<int32_t> tb = make_tokens(8000, 32);
  a.insert(a.end(), ta.begin(), ta.end());
  b.insert(b.end(), tb.begin(), tb.end());

  insert(c, a, 1);
  insert(c, b, 2);
  check_eq(ts_prefix_cache_node_count(c), 6, "2 shared nodes + 2 private each");

  /* Sequence 1 finishes. The shared prefix must NOT go with it. */
  ts_prefix_cache_evict(c, 1);
  check_eq(ts_prefix_cache_node_count(c), 4, "only seq 1's private blocks are freed");

  int32_t donor = TS_PREFIX_NO_SEQ;
  check_eq(match(c, shared, &donor), 32, "shared prefix still resident");
  check_eq(donor, 2, "donor rolls over to the surviving owner");

  /* And seq 1's private tail is genuinely gone. */
  check_eq(match(c, a, NULL), 32, "evicted sequence's private blocks are gone");

  ts_prefix_cache_evict(c, 2);
  check_eq(ts_prefix_cache_node_count(c), 0, "last owner leaving reclaims everything");
  check_eq(ts_prefix_cache_seq_count(c), 0, "no sequences left");
  ts_prefix_cache_destroy(c);
}

static void test_reinsert_supersedes_previous_claim() {
  std::printf("test_reinsert_supersedes_previous_claim\n");
  ts_prefix_cache *c = ts_prefix_cache_create(16);
  std::vector<int32_t> shortp = make_tokens(1000, 32);
  std::vector<int32_t> longp = make_tokens(1000, 64);

  insert(c, shortp, 5);
  check_eq(ts_prefix_cache_node_count(c), 2, "2 nodes for the short claim");
  insert(c, longp, 5); /* same sequence, now longer */
  check_eq(ts_prefix_cache_node_count(c), 4, "grown claim replaces, not duplicates");
  check_eq(ts_prefix_cache_seq_count(c), 1, "still one sequence");

  ts_prefix_cache_evict(c, 5);
  check_eq(ts_prefix_cache_node_count(c), 0, "superseded claim left nothing behind");
  ts_prefix_cache_destroy(c);
}

static void test_donor_is_most_recent_owner() {
  std::printf("test_donor_is_most_recent_owner\n");
  ts_prefix_cache *c = ts_prefix_cache_create(16);
  std::vector<int32_t> p = make_tokens(1000, 32);
  insert(c, p, 3);
  insert(c, p, 4);
  insert(c, p, 9);
  int32_t donor = TS_PREFIX_NO_SEQ;
  check_eq(match(c, p, &donor), 32, "match across three owners");
  check_eq(donor, 9, "most recent owner is the donor");

  ts_prefix_cache_evict(c, 9);
  check_eq(match(c, p, &donor), 32, "still resident");
  check_eq(donor, 4, "donor falls back to the next most recent");
  ts_prefix_cache_destroy(c);
}

static void test_deep_tree_teardown_does_not_recurse() {
  std::printf("test_deep_tree_teardown_does_not_recurse\n");
  /* 200k tokens at block_size 1 -> a 200k-deep chain. A recursive destructor
   * blows the stack here; the iterative one must not. */
  ts_prefix_cache *c = ts_prefix_cache_create(1);
  std::vector<int32_t> deep = make_tokens(0, 200000);
  check_eq(insert(c, deep, 1), 200000, "deep chain inserted");
  check_eq(ts_prefix_cache_node_count(c), 200000, "one node per token");
  ts_prefix_cache_destroy(c); /* the assertion is that we reach the next line */
  std::printf("  (survived teardown)\n");
}

static void test_degenerate_inputs() {
  std::printf("test_degenerate_inputs\n");
  check(ts_prefix_cache_create(0) == NULL, "block_size 0 rejected", 0, 0);
  check(ts_prefix_cache_create(-4) == NULL, "negative block_size rejected", 0, 0);
  ts_prefix_cache_destroy(NULL); /* must not crash */

  ts_prefix_cache *c = ts_prefix_cache_create(16);
  check_eq(ts_prefix_cache_match(c, NULL, 64, NULL), 0, "NULL tokens matches nothing");
  check_eq(ts_prefix_cache_insert(c, NULL, 64, 1), 0, "NULL tokens inserts nothing");
  ts_prefix_cache_evict(c, 999); /* unknown seq is a no-op */
  check_eq(ts_prefix_cache_node_count(c), 0, "still empty");
  ts_prefix_cache_destroy(c);
}

int main() {
  std::printf("tinyserve prefix cache -- ABI %s\n\n", ts_prefix_cache_abi_version());

  test_empty_cache_never_matches();
  test_exact_and_partial_overlap();
  test_matches_are_block_aligned();
  test_shared_prefix_survives_one_eviction();
  test_reinsert_supersedes_previous_claim();
  test_donor_is_most_recent_owner();
  test_deep_tree_teardown_does_not_recurse();
  test_degenerate_inputs();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
