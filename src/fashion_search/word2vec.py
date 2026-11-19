"""Word2Vec cosine scoring for conjunctive candidates.

A Word2Vec model is trained on the catalogue itself, on the same processed
terms as the index, so every query term the index knows has a vector and
domain words (kurta, jogger, puma) are learned from how products use them.
A text is represented by the average of its word vectors; a product's score is
the cosine similarity between its vector and the query's.
"""

import zlib
from collections.abc import Collection, Sequence
from functools import lru_cache

import numpy as np
from gensim.models import Word2Vec

from fashion_search.corpus import TEXT_FIELDS, all_terms
from fashion_search.ranking import SearchEngine

# Training settings. One worker and a fixed seed make training reproducible.
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2  # words seen once get no reliable vector
SKIP_GRAM = 1  # skip-gram learns rare words better than CBOW on a small corpus
EPOCHS = 10
SEED = 42


def _stable_hash(text: str) -> int:
    # Gensim seeds each word's initial vector with Python's hash(), which changes
    # between runs; a fixed hash keeps the trained vectors the same every time.
    return zlib.crc32(text.encode())


def train_model(engine: SearchEngine) -> Word2Vec:
    """Train Word2Vec on the catalogue, one sentence per product text field."""
    sentences = [
        list(terms)
        for field in TEXT_FIELDS
        for terms in engine.corpus[f"{field}_terms"]
        if len(terms) > 0
    ]
    return Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SKIP_GRAM,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        hashfxn=_stable_hash,
    )


@lru_cache(maxsize=4)
def model_for(engine: SearchEngine) -> Word2Vec:
    """The Word2Vec model of an engine's catalogue, trained once and cached."""
    return train_model(engine)


def text_vector(model: Word2Vec, terms: Sequence[str]) -> np.ndarray:
    """Average vector of the terms the model knows; a zero vector if it knows none."""
    known = [t for t in terms if t in model.wv]
    if not known:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean([model.wv[t] for t in known], axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, defined as 0 when either vector is zero."""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / norm if norm else 0.0


@lru_cache(maxsize=4)
def _product_vectors(engine: SearchEngine) -> dict[str, np.ndarray]:
    # Filled lazily: only products that appear as candidates get a vector.
    return {}


def product_vector(engine: SearchEngine, pid: str) -> np.ndarray:
    """Average vector of all of a product's indexed terms, cached per engine."""
    cache = _product_vectors(engine)
    if pid not in cache:
        cache[pid] = text_vector(model_for(engine), all_terms(engine.corpus.loc[pid]))
    return cache[pid]


def score_candidates(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
) -> dict[str, float]:
    """Return one Word2Vec cosine score per candidate product ID."""
    query = text_vector(model_for(engine), query_terms)
    return {pid: cosine(query, product_vector(engine, pid)) for pid in candidates}
