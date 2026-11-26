"""The search engine's web app: search page, results with an LLM summary,
product pages and an analytics dashboard.

Run it with `uv run python -m fashion_search.web` and open http://127.0.0.1:8088.
"""

import os
import uuid
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

from fashion_search import rag
from fashion_search.analytics import AnalyticsStore
from fashion_search.search import DEFAULT_METHOD, METHODS, Searcher

# The Part 2 test queries, offered as examples on the search page.
EXAMPLES = ["women printed kurta cotton", "men polo tshirt regular fit", "slim fit jogger men",
            "puma men tshirt", "women denim jacket"]
PAGES = {"index", "search", "product", "dashboard"}  # page views worth logging (not static files)


def create_app(searcher: Searcher | None = None, analytics: AnalyticsStore | None = None) -> Flask:
    """Build the app. Tests and the notebook can pass their own searcher and store."""
    load_dotenv()
    app = Flask(__name__)
    # A fixed SECRET_KEY keeps sessions across restarts; a random one is fine for development.
    app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
    app.searcher = searcher or Searcher()
    app.analytics = analytics or AnalyticsStore()
    app.jinja_env.filters.update(inr=_inr, image_size=_image_size)

    @app.before_request
    def start_session():
        session.setdefault("id", uuid.uuid4().hex)

    @app.after_request
    def log_request(response):
        if request.endpoint in PAGES:
            app.analytics.log_request(
                session_id=session["id"],
                method=request.method,
                path=request.full_path.rstrip("?"),
                status=response.status_code,
                user_agent=request.headers.get("User-Agent", ""),
                ip=request.remote_addr,
                referrer=request.referrer,
            )
        return response

    @app.get("/")
    def index():
        return render_template("index.html", methods=METHODS, method=DEFAULT_METHOD, examples=EXAMPLES)

    @app.get("/search")
    def search():
        query = request.args.get("q", "").strip()
        if not query:
            return redirect(url_for("index"))
        method = request.args.get("method", DEFAULT_METHOD)
        page = request.args.get("page", 1, type=int)
        found = app.searcher.search(query, method=method, page=page)
        if found.page > found.pages:  # an old or edited link past the last page
            return redirect(url_for("search", q=query, method=found.method, page=found.pages))
        query_id = app.analytics.log_query(
            session_id=session["id"],
            text=query,
            terms=found.terms,
            method=found.method,
            page=found.page,
            results=found.total,
            relaxed=found.relaxed,
            elapsed_ms=found.elapsed_ms,
        )
        return render_template(
            "results.html",
            found=found,
            query_id=query_id,
            methods=METHODS,
            method=found.method,
            summary_enabled=rag.available() and found.total > 0,
        )

    @app.get("/summary")
    def summary():
        """The LLM summary of a query's first page, fetched by the results page after it loads."""
        query = request.args.get("q", "").strip()
        found = app.searcher.search(query, method=request.args.get("method", DEFAULT_METHOD))
        try:
            result = rag.summarize(query, found.results)
        except rag.SummaryUnavailable as error:  # rate limit, wrong key, no connection...
            app.logger.warning("RAG summary failed: %s (%s)", error, error.__cause__)
            return jsonify(text=None, error=str(error))
        except Exception:  # anything unexpected must not break the results page
            app.logger.exception("RAG summary failed")
            return jsonify(text=None, error="The summary is not available right now.")
        if result is None:
            return jsonify(text=None, error="No summary for this search.")
        return jsonify(text=result.text, variant=result.variant, model=result.model)

    @app.get("/product/<pid>")
    def product(pid: str):
        record = app.searcher.product(pid)
        if record is None:
            abort(404)
        app.analytics.log_click(
            session_id=session["id"],
            pid=pid,
            query_id=request.args.get("qid", type=int),
            rank=request.args.get("rank", type=int),
        )
        back = request.referrer if _is_results_page(request.referrer) else None
        return render_template("product.html", product=record, back=back)

    @app.get("/dashboard")
    def dashboard():
        # TODO(analytics): replace with the real dashboard (indicators, graphs, explanations).
        return render_template("dashboard.html", summary=app.analytics.summary())

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    return app


def _is_results_page(url: str | None) -> bool:
    return bool(url) and urlparse(url).path == "/search" and urlparse(url).netloc == request.host


def _inr(value) -> str:
    """Format a price in rupees: 1299.0 -> ₹1,299."""
    return f"₹{value:,.0f}" if value is not None else ""


def _image_size(url: str, size: int) -> str:
    """The same catalogue image at another size (the dataset links 128 px thumbnails)."""
    return url.replace("/image/128/128/", f"/image/{size}/{size}/")
