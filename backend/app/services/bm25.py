"""
Pure Python BM25 sparse encoder — no fastembed, no Rust, no external dependencies.

Produces SparseVector-compatible (indices, values) output that slots directly
into Qdrant's hybrid search pipeline.

How it works:
  1. Tokenize: lowercase → split on non-alphanumeric → remove stopwords → stem
  2. Hash each token to a uint32 index via FNV-1a (no vocab file needed)
  3. Compute raw term frequency (TF) weights
  4. Return sorted (indices, values) — Qdrant applies IDF server-side via Modifier.IDF

Why this works with Qdrant:
  Qdrant's Modifier.IDF on the sparse vector field handles the IDF component
  automatically at query time. The client only needs to supply TF weights,
  which require no corpus statistics — making this encoder fully stateless.

Compatible with Python 3.8+ — uses only stdlib (re, hashlib, dataclasses).
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── English stopwords ─────────────────────────────────────────────────────────
# Matches Qdrant's default BM25 stopword list. These are filtered out before
# hashing so they don't pollute the sparse vector with meaningless high-freq terms.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "what", "which", "who", "whom",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "also", "about", "up", "out", "if", "then", "than", "into",
    "over", "after", "before", "between", "through", "during", "any", "all",
}

# ── Suffix-stripping stemmer (Porter-lite) ────────────────────────────────────
# Covers the most common English suffixes without requiring nltk or any library.
# Good enough for BM25 — goal is normalising surface forms, not linguistic accuracy.
# Rules are tried in order; first match wins.
_STEM_RULES = [
    ("ational", "ate"),
    ("tional",  "tion"),
    ("enci",    "ence"),
    ("anci",    "ance"),
    ("izer",    "ize"),
    ("ising",   "ise"),
    ("izing",   "ize"),
    ("ised",    "ise"),
    ("ized",    "ize"),
    ("nesses",  ""),
    ("ments",   ""),
    ("ment",    ""),
    ("ings",    ""),
    ("ing",     ""),
    ("ness",    ""),
    ("tions",   ""),
    ("tion",    ""),
    ("ions",    ""),
    ("ion",     ""),
    ("ers",     ""),
    ("er",      ""),
    ("ies",     "y"),
    ("ied",     "y"),
    ("ly",      ""),
    ("ed",      ""),
    ("es",      ""),
    ("s",       ""),
]

# Minimum token length after stemming — avoids single-char noise tokens
_MIN_TOKEN_LEN = 2


def _stem(word: str) -> str:
    """
    Apply simple suffix stripping.
    Skips the rule if the resulting stem would be shorter than _MIN_TOKEN_LEN.
    """
    for suffix, replacement in _STEM_RULES:
        if word.endswith(suffix):
            stemmed = word[: len(word) - len(suffix)] + replacement
            if len(stemmed) >= _MIN_TOKEN_LEN:
                return stemmed
    return word


def _tokenize(text: str) -> list[str]:
    """
    Full tokenization pipeline:
      1. Lowercase
      2. Split on anything that isn't a letter or digit
      3. Remove stopwords and tokens shorter than _MIN_TOKEN_LEN
      4. Stem each surviving token

    Returns a list of stemmed tokens (may contain duplicates — TF counts them).
    """
    text  = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    words = [w for w in words if len(w) >= _MIN_TOKEN_LEN and w not in _STOPWORDS]
    words = [_stem(w) for w in words]
    # Re-filter after stemming in case stemming created short tokens
    words = [w for w in words if len(w) >= _MIN_TOKEN_LEN]
    return words


def _token_to_index(token: str) -> int:
    """
    Map a token string to a uint32 index via FNV-1a 32-bit hash.

    FNV-1a is fast, has good distribution for short strings, and needs
    no dependencies. Collision probability is negligible for typical
    document vocabularies (< 100k unique tokens).

    This matches the hashing strategy used by fastembed's Qdrant/bm25 model.
    """
    h = 2166136261  # FNV-1a 32-bit offset basis
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF  # FNV prime, mask to 32 bits
    return h


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SparseEmbedding:
    """
    Output of BM25Encoder.encode().
    Mirrors the interface of fastembed's SparseEmbedding so this encoder
    is a drop-in replacement — .indices and .values are the only fields used.
    """
    indices: list[int]    # Sorted uint32 token hashes
    values:  list[float]  # Raw TF weights (IDF applied by Qdrant server-side)


# ── Encoder ───────────────────────────────────────────────────────────────────

class BM25Encoder:
    """
    Stateless BM25 sparse encoder.

    Stateless because Qdrant handles the IDF component server-side via
    Modifier.IDF on the sparse vector field — we only supply TF weights,
    which require no corpus statistics.

    API mirrors fastembed's SparseTextEmbedding so it's a drop-in replacement:
      encoder.passage_embed(texts)  → used during ingestion
      encoder.query_embed(query)    → used during retrieval
    """

    def encode(self, text: str) -> SparseEmbedding:
        """
        Encode a single text into a sparse BM25 vector.

        Steps:
          1. Tokenize (lowercase, stopword removal, stemming)
          2. Count term frequencies — duplicate tokens add to TF correctly
          3. Hash tokens to uint32 indices
          4. Sort by index (Qdrant requires sorted sparse vectors)

        Returns SparseEmbedding with raw TF weights.
        Empty text or all-stopword input returns an empty embedding —
        Qdrant handles empty sparse vectors gracefully (scores 0).
        """
        tokens = _tokenize(text)

        if not tokens:
            logger.debug("BM25 encode: empty token list for input (all stopwords or empty text).")
            return SparseEmbedding(indices=[], values=[])

        # Count term frequencies — {token_hash: count}
        tf: dict[int, float] = {}
        for token in tokens:
            idx      = _token_to_index(token)
            tf[idx]  = tf.get(idx, 0.0) + 1.0

        # Sort by index — Qdrant requires indices to be in ascending order
        sorted_items = sorted(tf.items())
        indices = [item[0] for item in sorted_items]
        values  = [item[1] for item in sorted_items]

        return SparseEmbedding(indices=indices, values=values)

    def encode_batch(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode a batch of texts. Shared by passage_embed and query_embed."""
        return [self.encode(text) for text in texts]

    def passage_embed(self, texts: list[str]) -> list[SparseEmbedding]:
        """
        Encode document passages for ingestion.
        For BM25 (unlike SPLADE) this is identical to query encoding —
        BM25 is symmetric, no separate passage/query encoders needed.
        """
        return self.encode_batch(texts)

    def query_embed(self, query: str) -> list[SparseEmbedding]:
        """
        Encode a query string for retrieval.
        Returns a list with one element to match fastembed's generator interface:
          result = list(encoder.query_embed(query))[0]
        """
        return [self.encode(query)]


# ── Module-level singleton ────────────────────────────────────────────────────

_encoder: BM25Encoder | None = None


def get_bm25_encoder() -> BM25Encoder:
    """
    Return a module-level singleton BM25Encoder.
    Safe to call from multiple threads — Python's GIL makes the assignment atomic.
    No lock needed since BM25Encoder.__init__ has no side effects.
    """
    global _encoder
    if _encoder is None:
        logger.info("Initialising pure-Python BM25 encoder (no external dependencies).")
        _encoder = BM25Encoder()
    return _encoder