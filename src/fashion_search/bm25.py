"""BM25 scoring for conjunctive candidates.

Okapi BM25 scores a product as the sum, over the distinct query terms, of

    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

where tf is how often t appears in the product, dl is the product's length
(its number of indexed terms) and avgdl the average length. Unlike TF-IDF, the
term frequency saturates (k1 bounds what repeating a word can add) and length
is normalised against the average product (b sets how strongly).

IDF uses the Robertson-Sparck Jones form with 1 added inside the logarithm,
log(1 + (N - df + 0.5) / (df + 0.5)), as Lucene does. Without the 1, terms in
more than half the products (wear, wash, cotton) would get a negative IDF and
lower the score of every product that contains them.
"""

import math
from collections.abc import Collection, Sequence
from functools import lru_cache

from fashion_search.ranking import SearchEngine

K1 = 1.2  # term-frequency saturation; the usual default (range 1.2 to 2.0)
B = 0.75  # length normalisation, from 0 (none) to 1 (full); the usual default


@lru_cache(maxsize=4)
def corpus_statistics(engine: SearchEngine) -> tuple[dict[str, int], float]:
    """Each product's length (number of indexed terms) and the average length.

    Cached per engine, since they only depend on the index.
    """
    lengths: dict[str, int] = {}
    for posting in engine.index.values():
        for pid, tf in posting.items():
            lengths[pid] = lengths.get(pid, 0) + tf
    return lengths, sum(lengths.values()) / len(lengths)


def idf(engine: SearchEngine, term: str) -> float:
    """Robertson-Sparck Jones IDF with 1 added inside the log, so it is never negative."""
    n_docs = len(engine.corpus)
    df = len(engine.index.get(term, {}))
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def score_candidates(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
    k1: float = K1,
    b: float = B,
) -> dict[str, float]:
    """Return one BM25 score per candidate product ID."""
    lengths, avg_length = corpus_statistics(engine)
    terms = set(query_terms)  # a repeated query word counts once
    idfs = {t: idf(engine, t) for t in terms}

    scores = {}
    for pid in candidates:
        norm = k1 * (1 - b + b * lengths[pid] / avg_length)
        score = 0.0
        for t in terms:
            tf = engine.index.get(t, {}).get(pid, 0)
            score += idfs[t] * tf * (k1 + 1) / (tf + norm)
        scores[pid] = score
    return scores
