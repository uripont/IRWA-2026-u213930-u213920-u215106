"""Usage data of the search engine, kept in memory.

The web app (fashion_search.web) calls the `log_*` methods: one request row
per page request, one query row per results page shown and one click row per
product page opened (with the query and rank when it came from a results page). Rows are plain dataclasses in lists, so the dashboard and the
notebook can turn them into DataFrames (`AnalyticsStore.table`).

This is the starting point of the analytics workstream. What the brief still
asks for is marked TODO(analytics) below.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd

# TODO(analytics): design the data model (a star schema, e.g. a session
# dimension, plus request, query and click facts) and replace these rows.
# TODO(analytics): parse the user agent (browser, OS, device) and add time of
# day, and optionally country and city from the IP.
# TODO(analytics): dwell time (time between opening a product and coming back
# to the results), physical sessions and, optionally, missions.
# TODO(analytics): the dashboard page (indicators, graphs and their explanations).


@dataclass
class RequestRow:
    id: int
    session_id: str
    timestamp: datetime
    method: str
    path: str
    status: int
    user_agent: str
    ip: str | None
    referrer: str | None


@dataclass
class QueryRow:
    id: int
    session_id: str
    timestamp: datetime
    text: str
    terms: list[str]
    method: str
    page: int
    results: int  # products found, over all pages
    relaxed: bool  # True when no product had every query term
    elapsed_ms: float


@dataclass
class ClickRow:
    id: int
    session_id: str
    timestamp: datetime
    query_id: int | None  # None when the product was opened without a search
    pid: str
    rank: int | None


@dataclass
class AnalyticsStore:
    requests: list[RequestRow] = field(default_factory=list)
    queries: list[QueryRow] = field(default_factory=list)
    clicks: list[ClickRow] = field(default_factory=list)

    def log_request(self, session_id: str, method: str, path: str, status: int,
                    user_agent: str, ip: str | None, referrer: str | None) -> int:
        row = RequestRow(len(self.requests) + 1, session_id, datetime.now(), method, path,
                         status, user_agent, ip, referrer)
        self.requests.append(row)
        return row.id

    def log_query(self, session_id: str, text: str, terms: list[str], method: str, page: int,
                  results: int, relaxed: bool, elapsed_ms: float) -> int:
        row = QueryRow(len(self.queries) + 1, session_id, datetime.now(), text, terms, method,
                       page, results, relaxed, elapsed_ms)
        self.queries.append(row)
        return row.id

    def log_click(self, session_id: str, pid: str, query_id: int | None, rank: int | None) -> int:
        row = ClickRow(len(self.clicks) + 1, session_id, datetime.now(), query_id, pid, rank)
        self.clicks.append(row)
        return row.id

    def table(self, name: str) -> pd.DataFrame:
        """One of the tables ("requests", "queries", "clicks") as a DataFrame."""
        return pd.DataFrame([asdict(row) for row in getattr(self, name)])

    def summary(self) -> dict[str, int]:
        """Headline counts for the dashboard."""
        # TODO(analytics): define the key indicators, metrics and reports.
        return {
            "requests": len(self.requests),
            "sessions": len({r.session_id for r in self.requests}),
            "queries": len(self.queries),
            "clicks": len(self.clicks),
        }
