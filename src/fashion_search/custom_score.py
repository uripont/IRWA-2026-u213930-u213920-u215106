"""Custom relevance scoring for conjunctive candidates.

A product's score multiplies how well its text matches the query by how good
an offer it is:

    score = (W_TEXT * text + W_NAME * name)
            * (BASE + W_RATING * rating + W_DISCOUNT * discount)
            * (OUT_OF_STOCK if the product is out of stock else 1)

- text: the product's BM25 score divided by the best BM25 score among the
  candidates, so it lies in (0, 1] whatever the query.
- name: the share of distinct query terms found in the product's title or
  brand. Every candidate contains every query term somewhere, but a jogger
  whose title says "jogger" is a better match than a track pant that mentions
  joggers in its description.
- rating: the average rating scaled from 1-5 to 0-1. Products without a rating
  get the catalogue's average, so they are neither rewarded nor punished.
- discount: the discount as a fraction (0.53 for 53% off), 0 when missing.

The offer factor ranges from BASE to 1, so text relevance always decides most
of the order and rating and discount mostly reorder products with similar
text. Out-of-stock products stay in the results but move down, since a
user cannot buy them. Price is left out on purpose: whether a cheaper product
is better depends on the user, and the discount already captures a good deal.
"""

from collections.abc import Collection, Sequence
from functools import lru_cache

import pandas as pd

from fashion_search import bm25
from fashion_search.ranking import SearchEngine

W_TEXT = 0.6  # weight of the BM25 text match
W_NAME = 0.4  # weight of the query terms found in the title or brand
BASE = 0.8  # offer factor of a product with the lowest rating and no discount
W_RATING = 0.15  # what a 5-star rating adds to the offer factor
W_DISCOUNT = 0.05  # what a 100% discount would add to the offer factor
OUT_OF_STOCK = 0.5  # factor applied to products that cannot be bought


@lru_cache(maxsize=4)
def mean_rating(engine: SearchEngine) -> float:
    """Average rating over the rated products of the catalogue."""
    return float(engine.corpus["average_rating"].mean())


def features(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
) -> pd.DataFrame:
    """Each candidate's score components, all scaled to 0-1, indexed by product ID."""
    terms = set(query_terms)
    pids = sorted(candidates)
    if not pids:
        return pd.DataFrame(columns=["text", "name", "rating", "discount", "in_stock"])
    records = engine.corpus.loc[pids]

    text = pd.Series(bm25.score_candidates(engine, query_terms, pids))[pids]
    best = text.max()
    name_terms = [set(title) | set(brand) for title, brand in zip(records["title_terms"], records["brand_terms"])]

    return pd.DataFrame({
        "text": (text / best if best > 0 else 1.0).values,
        "name": [len(terms & name) / len(terms) for name in name_terms],
        "rating": ((records["average_rating"].fillna(mean_rating(engine)) - 1) / 4).astype(float).values,
        "discount": (records["discount"].fillna(0) / 100).astype(float).values,
        "in_stock": (~records["out_of_stock"]).values,
    }, index=pd.Index(pids, name="pid"))


def combine(
    table: pd.DataFrame,
    w_text: float = W_TEXT,
    w_name: float = W_NAME,
    base: float = BASE,
    w_rating: float = W_RATING,
    w_discount: float = W_DISCOUNT,
    out_of_stock: float = OUT_OF_STOCK,
) -> pd.Series:
    """The custom score of each row of a features() table.

    The weights default to the module's, and can be changed to see what each
    component does to the ranking.
    """
    relevance = w_text * table["text"] + w_name * table["name"]
    offer = base + w_rating * table["rating"] + w_discount * table["discount"]
    stock = table["in_stock"].map({True: 1.0, False: out_of_stock}).astype(float)
    return relevance * offer * stock


def score_candidates(
    engine: SearchEngine,
    query_terms: Sequence[str],
    candidates: Collection[str],
) -> dict[str, float]:
    """Return one custom score per candidate product ID."""
    return combine(features(engine, query_terms, candidates)).to_dict()
