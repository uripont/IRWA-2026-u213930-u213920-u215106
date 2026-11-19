"""Word2Vec cosine scoring for conjunctive candidates."""

from collections.abc import Collection, Sequence

from fashion_search.ranking import SearchEngine


def score_candidates(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
) -> dict[str, float]:
    """Return one score per candidate product ID."""
    # TODO: Prepare the model and cache averaged document vectors.
    # TODO: Average query vectors and define out-of-vocabulary and zero-vector handling.
    raise NotImplementedError("Part 3 word2vec scoring is pending.")
