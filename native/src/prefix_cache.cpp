/* Block-aligned radix tree over token IDs. See include/tinyserve/prefix_cache.h
 * for why this layer is C++ rather than Python.
 *
 * SHAPE OF THE TREE
 * -----------------
 * One node per *block* of block_size tokens, not one node per token. The edge
 * key is the block's token vector, so a path from the root spells out a
 * block-aligned prompt prefix and depth in blocks is exactly the match length
 * divided by block_size.
 *
 * Blocking the tree this way is what makes the result directly actionable: a
 * token-granular radix tree needs node splitting on partial matches and then
 * reports matches that do not land on a KV block boundary, which the caller
 * cannot use. Here the structure and llama.cpp's KV block accounting agree by
 * construction, and there is no split path to get wrong.
 *
 * OWNERSHIP
 * ---------
 * A sequence is recorded as an owner of *every* node along its path, not just
 * the deepest. That costs one int32 per block per sequence -- the same order as
 * the token vector it describes -- and buys two properties worth more than the
 * memory:
 *
 *   1. Every live node has at least one owner, so a match can always name a
 *      donor. Recording only the tail leaves interior nodes ownerless and the
 *      donor lookup has to search the subtree for a live descendant.
 *   2. Eviction is complete without an upward cascade. All ancestors of a node
 *      owned by S are themselves owned by S, so S's own node list is closed
 *      under "parent of". Walking it deepest-first frees exactly what died and
 *      cannot leave a dangling pointer, because each node is visited once.
 */

#include "tinyserve/prefix_cache.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <vector>

namespace {

typedef std::vector<int32_t> Block;

struct Node {
  Node *parent;
  Block key;                      /* this block's tokens; empty at the root */
  std::map<Block, Node *> children;
  std::vector<int32_t> owners;    /* insertion-ordered, unique */
  int32_t depth;                  /* in blocks; 0 at the root */

  Node() : parent(NULL), depth(0) {}
};

void drop_owner(std::vector<int32_t> &owners, int32_t seq_id) {
  std::vector<int32_t>::iterator it = std::find(owners.begin(), owners.end(), seq_id);
  if (it != owners.end()) {
    owners.erase(it);
  }
}

/* Sort helper: deepest first. Ties are irrelevant -- two nodes at the same depth
 * are never ancestor and descendant of one another. */
struct DeeperFirst {
  bool operator()(const Node *a, const Node *b) const { return a->depth > b->depth; }
};

}  /* namespace */

struct ts_prefix_cache {
  int32_t block_size;
  Node root;
  std::unordered_map<int32_t, std::vector<Node *> > seq_nodes;
  int32_t node_count; /* excludes the root */

  explicit ts_prefix_cache(int32_t bs) : block_size(bs), node_count(0) {}
};

extern "C" {

ts_prefix_cache *ts_prefix_cache_create(int32_t block_size) {
  if (block_size < 1) {
    return NULL;
  }
  return new ts_prefix_cache(block_size);
}

void ts_prefix_cache_destroy(ts_prefix_cache *cache) {
  if (cache == NULL) {
    return;
  }
  /* Iterative post-order teardown. Recursion here would be bounded by prompt
   * length in blocks, which is caller-controlled -- a deep enough tree would
   * overflow the stack, and "long prompt crashes the server" is not an
   * acceptable failure mode for an untrusted input path. */
  std::vector<Node *> stack;
  for (std::map<Block, Node *>::iterator it = cache->root.children.begin();
       it != cache->root.children.end(); ++it) {
    stack.push_back(it->second);
  }
  while (!stack.empty()) {
    Node *node = stack.back();
    stack.pop_back();
    for (std::map<Block, Node *>::iterator it = node->children.begin();
         it != node->children.end(); ++it) {
      stack.push_back(it->second);
    }
    delete node;
  }
  delete cache;
}

int32_t ts_prefix_cache_match(ts_prefix_cache *cache, const int32_t *tokens, int32_t n_tokens,
                              int32_t *out_seq_id) {
  if (out_seq_id != NULL) {
    *out_seq_id = TS_PREFIX_NO_SEQ;
  }
  if (cache == NULL || tokens == NULL || n_tokens < cache->block_size) {
    return 0;
  }

  const int32_t bs = cache->block_size;
  const int32_t n_blocks = n_tokens / bs;
  Node *node = &cache->root;
  int32_t matched_blocks = 0;

  for (int32_t b = 0; b < n_blocks; ++b) {
    Block key(tokens + (std::size_t)b * bs, tokens + (std::size_t)(b + 1) * bs);
    std::map<Block, Node *>::iterator it = node->children.find(key);
    if (it == node->children.end()) {
      break;
    }
    node = it->second;
    ++matched_blocks;
  }

  if (matched_blocks == 0) {
    return 0;
  }
  /* Every live node has an owner (see the ownership note at the top), so this
   * is not a "should never happen" branch being defensive -- it is an invariant
   * the eviction path is responsible for maintaining. */
  if (out_seq_id != NULL && !node->owners.empty()) {
    /* Most recent owner: the sequence most likely to still hold these cells. */
    *out_seq_id = node->owners.back();
  }
  return matched_blocks * bs;
}

int32_t ts_prefix_cache_insert(ts_prefix_cache *cache, const int32_t *tokens, int32_t n_tokens,
                               int32_t seq_id) {
  if (cache == NULL || tokens == NULL || n_tokens < cache->block_size) {
    return 0;
  }

  /* A sequence holds exactly one claim. Re-inserting supersedes the old path
   * rather than adding a second one, so a sequence that grows as it decodes
   * does not leave its shorter self behind as a phantom donor. */
  ts_prefix_cache_evict(cache, seq_id);

  const int32_t bs = cache->block_size;
  const int32_t n_blocks = n_tokens / bs;
  Node *node = &cache->root;
  std::vector<Node *> &owned = cache->seq_nodes[seq_id];
  owned.reserve((std::size_t)n_blocks);

  for (int32_t b = 0; b < n_blocks; ++b) {
    Block key(tokens + (std::size_t)b * bs, tokens + (std::size_t)(b + 1) * bs);
    std::map<Block, Node *>::iterator it = node->children.find(key);
    if (it == node->children.end()) {
      Node *child = new Node();
      child->parent = node;
      child->key = key;
      child->depth = node->depth + 1;
      node->children[key] = child;
      cache->node_count += 1;
      node = child;
    } else {
      node = it->second;
    }
    node->owners.push_back(seq_id);
    owned.push_back(node);
  }

  return n_blocks * bs;
}

void ts_prefix_cache_evict(ts_prefix_cache *cache, int32_t seq_id) {
  if (cache == NULL) {
    return;
  }
  std::unordered_map<int32_t, std::vector<Node *> >::iterator entry =
      cache->seq_nodes.find(seq_id);
  if (entry == cache->seq_nodes.end()) {
    return;
  }

  std::vector<Node *> owned = entry->second;
  cache->seq_nodes.erase(entry);

  /* Deepest first, so a node is only tested for deletion after every child of
   * it that this sequence owned has already been unlinked. A node still held by
   * another sequence, or still carrying a surviving child, stays -- which is
   * what keeps one sequence finishing from tearing a prefix out from under
   * another that is still sharing it. */
  std::sort(owned.begin(), owned.end(), DeeperFirst());

  for (std::size_t i = 0; i < owned.size(); ++i) {
    Node *node = owned[i];
    drop_owner(node->owners, seq_id);
    if (!node->owners.empty() || !node->children.empty()) {
      continue;
    }
    if (node->parent != NULL) {
      node->parent->children.erase(node->key);
    }
    delete node;
    cache->node_count -= 1;
  }
}

int32_t ts_prefix_cache_node_count(const ts_prefix_cache *cache) {
  return cache == NULL ? 0 : cache->node_count;
}

int32_t ts_prefix_cache_seq_count(const ts_prefix_cache *cache) {
  return cache == NULL ? 0 : (int32_t)cache->seq_nodes.size();
}

const char *ts_prefix_cache_abi_version(void) { return "1.0"; }

}  /* extern "C" */
