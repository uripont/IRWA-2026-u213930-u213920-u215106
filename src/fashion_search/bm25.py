"""BM25 scoring for conjunctive candidates."""

from collections.abc import Collection, Sequence

from fashion_search.ranking import SearchEngine


def score_candidates(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
) -> dict[str, float]:
    """Return one score per candidate product ID."""
    # TODO: Precompute document lengths and average length from the indexed terms.
    # TODO: Choose the IDF convention and expose k1 and b.
    raise NotImplementedError("Part 3 bm25 scoring is pending.")
