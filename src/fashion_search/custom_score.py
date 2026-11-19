"""Custom relevance scoring for conjunctive candidates."""

from collections.abc import Collection, Sequence

from fashion_search.ranking import SearchEngine


def score_candidates(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
) -> dict[str, float]:
    """Return one score per candidate product ID."""
    # TODO: Define the formula and its weights.
    # TODO: Normalize the chosen catalogue fields and handle missing values.
    raise NotImplementedError("Part 3 custom_score scoring is pending.")
