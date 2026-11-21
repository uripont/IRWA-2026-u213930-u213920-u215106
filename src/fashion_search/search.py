"""The general search function: one query in, one page of results out.

`Searcher` loads the catalogue and index once and keeps every ranker from
Part 3 ready (TF-IDF, BM25, the custom score and Word2Vec), so the web app
only has to call `search`. Compared with the notebooks, it adds what a real
search box needs:

- Relaxed matching. AND matching returns nothing as soon as one query word is
  missing from every product (a typo, a word the catalogue never uses). Then
  the searcher keeps the products that contain the most query words, ranks
  them with the same method, and reports how many words they match.
- Caching. The full ranking of a query is computed once and reused for the
  next pages and for the RAG summary, which asks for the same results.
- Word2Vec only when needed. Its model is loaded (or trained) the first time a
  user picks it, so start-up takes a few seconds.

Usage:
    searcher = Searcher()
    page = searcher.search("men slim jeans blue", method="bm25", page=1)
"""

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from fashion_search import bm25, custom_score, word2vec
from fashion_search.corpus import RESULT_FIELDS, load_corpus
from fashion_search.index import match
from fashion_search.ranking import SearchEngine
from fashion_search.text import build_terms

# Ranking method name -> what the user sees. The custom score is the default
# since Part 3 found it the most useful for shopping (same quality as BM25,
# fewer out-of-stock products, better rated ones).
METHODS = {
    "custom": "Best match (custom score)",
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "word2vec": "Word2Vec",
}
DEFAULT_METHOD = "custom"
PER_PAGE = 20
CACHE_SIZE = 256  # queries whose full ranking is kept in memory

Scorer = Callable[[list[str], set[str]], dict[str, float]]


@dataclass(frozen=True)
class Result:
    """One product in a result list, with the fields a results page shows."""

    pid: str
    rank: int
    score: float
    title: str
    description: str | None
    brand: str | None
    category: str | None
    sub_category: str | None
    selling_price: float | None
    actual_price: float | None
    discount: float | None
    average_rating: float | None
    out_of_stock: bool
    url: str | None
    image: str | None


@dataclass
class SearchResults:
    """One page of results for a query, plus what the page needs to explain them."""

    query: str
    method: str
    terms: list[str]
    total: int  # products found, over all pages
    page: int
    per_page: int
    results: list[Result]
    # With relaxed matching: how many distinct query terms each result
    # contains, and the terms none of the results contains. None and empty
    # with normal AND matching (every result has every term).
    matched_terms: int | None = None
    dropped_terms: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def pages(self) -> int:
        return max(1, math.ceil(self.total / self.per_page))

    @property
    def relaxed(self) -> bool:
        return self.matched_terms is not None


class Searcher:
    """Search the catalogue with any of the Part 3 rankers."""

    def __init__(self, corpus: pd.DataFrame | None = None):
        self.engine = SearchEngine(load_corpus() if corpus is None else corpus)
        self.products = self.engine.corpus
        self._scorers: dict[str, Scorer] = {
            "tfidf": self.engine.score,
            "bm25": lambda terms, cands: bm25.score_candidates(self.engine, terms, cands),
            "custom": lambda terms, cands: custom_score.score_candidates(self.engine, terms, cands),
            "word2vec": lambda terms, cands: word2vec.score_candidates(self.engine, terms, cands),
        }
        self._ranked = lru_cache(maxsize=CACHE_SIZE)(self._rank)

    def search(
        self,
        query: str,
        method: str = DEFAULT_METHOD,
        page: int = 1,
        per_page: int = PER_PAGE,
    ) -> SearchResults:
        """One page of the products that best match the query, best first."""
        start = time.perf_counter()
        method = method if method in METHODS else DEFAULT_METHOD
        terms = build_terms(query)
        ranked, matched, dropped = self._ranked(tuple(terms), method)
        page = max(1, page)
        first = (page - 1) * per_page
        results = [
            self._result(pid, score, rank)
            for rank, (pid, score) in enumerate(ranked[first:first + per_page], start=first + 1)
        ]
        return SearchResults(
            query=query,
            method=method,
            terms=terms,
            total=len(ranked),
            page=page,
            per_page=per_page,
            results=results,
            matched_terms=matched,
            dropped_terms=list(dropped),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    def product(self, pid: str) -> dict | None:
        """A product's display fields (missing values as None), or None if the pid is unknown."""
        if pid not in self.products.index:
            return None
        record = self.products.loc[pid, RESULT_FIELDS].to_dict()
        return {k: None if pd.api.types.is_scalar(v) and pd.isna(v) else v for k, v in record.items()}

    def _rank(self, terms: tuple[str, ...], method: str) -> tuple[list[tuple[str, float]], int | None, tuple[str, ...]]:
        """All matching (pid, score) pairs best first, with how the candidates were matched.

        Returns the ranking, the number of terms each result contains (None
        for AND matching) and the terms no result contains.
        """
        candidates, matched = self._candidates(list(terms))
        distinct = list(dict.fromkeys(terms))
        if not candidates:
            return [], matched, tuple(distinct) if matched is not None else ()
        index = self.engine.index
        dropped = tuple(t for t in distinct if not any(pid in index.get(t, {}) for pid in candidates))
        kept = [t for t in terms if t not in dropped]
        scores = self._scorers[method](kept, candidates)
        return sorted(scores.items(), key=lambda s: (-s[1], s[0])), matched, dropped

    def _candidates(self, terms: list[str]) -> tuple[set[str], int | None]:
        """AND matches, or else the products with the most distinct query terms (and how many)."""
        matches = match(self.engine.index, terms)
        if matches or len(set(terms)) < 2:
            return matches, None
        counts = Counter(pid for t in set(terms) for pid in self.engine.index.get(t, {}))
        if not counts:
            return set(), 0
        best = max(counts.values())
        return {pid for pid, n in counts.items() if n == best}, best

    def _result(self, pid: str, score: float, rank: int) -> Result:
        p = self.products.loc[pid]
        images = p.get("images")
        return Result(
            pid=pid,
            rank=rank,
            score=score,
            title=p["title"],
            description=_text(p["description"]),
            brand=_text(p["brand"]),
            category=_text(p["category"]),
            sub_category=_text(p["sub_category"]),
            selling_price=_number(p["selling_price"]),
            actual_price=_number(p["actual_price"]),
            discount=_number(p["discount"]),
            average_rating=_number(p["average_rating"]),
            out_of_stock=bool(p["out_of_stock"]),
            url=_text(p["url"]),
            image=images[0] if images is not None and len(images) else None,
        )


def _number(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _text(value) -> str | None:
    return None if not isinstance(value, str) or not value else value
