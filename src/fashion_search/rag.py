"""Summaries of a result list written by an LLM (retrieval-augmented generation).

The search engine retrieves the products and an LLM, reached through the Groq
API as in the course template, writes a short recommendation from them. Each
way of building the prompt is a *variant*, so the baseline and the improved
versions can be compared on the same queries (see notebooks/part_4.ipynb).

Set GROQ_API_KEY (and optionally GROQ_MODEL) in .env, which is read on import.
Without a key, `summarize` returns None, so callers can simply leave the
summary out. When the API refuses or fails (the free tier's rate limit, a
wrong key, no connection), `summarize` raises `SummaryUnavailable` with a
message that can be shown to the user. Answers are cached per prompt, so the
same results are only sent to the API once.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from fashion_search.search import Result

load_dotenv()

# The course template used llama-3.1-8b-instant, which Groq no longer offers.
# gpt-oss-20b is the closest small, fast general model it serves now.
DEFAULT_MODEL = "openai/gpt-oss-20b"

# The course template's prompt, unchanged.
BASELINE_PROMPT = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Identify the single best product that matches the user's request.
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        3. If there is another product that could also work, mention it briefly as an alternative.
        4. If no product is a good fit, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """


TIMEOUT_S = 20  # seconds to wait for an answer before giving up
CACHE_SIZE = 256  # answers kept in memory


class SummaryUnavailable(Exception):
    """The API could not give a summary. The message is meant for the user."""


@dataclass(frozen=True)
class Summary:
    text: str
    variant: str
    model: str


def baseline_prompt(query: str, results: Sequence[Result], top_n: int = 20) -> str:
    """The template's prompt: only the pid and title of the top 20 products."""
    retrieved = "\n".join(f"- PID: {r.pid}, Title: {r.title}" for r in results[:top_n])
    return BASELINE_PROMPT.format(retrieved_results=retrieved, user_query=query)


# TODO(RAG): add the two or three improvements as new variants, each a function
# from (query, results) to a prompt, e.g. product metadata in the prompt
# (price, discount, rating, stock), a stricter prompt, or handling queries
# where no retrieved product fits. Changes to the call itself (temperature,
# model, a second retrieval pass) can go in `summarize` behind the variant name.
VARIANTS: dict[str, Callable[[str, Sequence[Result]], str]] = {
    "baseline": baseline_prompt,
}
DEFAULT_VARIANT = "baseline"  # TODO(RAG): switch to the best improved variant


def available() -> bool:
    """Whether an API key is configured."""
    return bool(os.environ.get("GROQ_API_KEY"))


def summarize(query: str, results: Sequence[Result], variant: str = DEFAULT_VARIANT) -> Summary | None:
    """An LLM summary of the results for the query, or None without an API key or results."""
    if not available() or not results:
        return None
    model = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    prompt = VARIANTS[variant](query, results)
    return Summary(text=complete(prompt, model), variant=variant, model=model)


@lru_cache(maxsize=CACHE_SIZE)  # errors are not cached, so a later call tries again
def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send one user message to the Groq API and return the reply's text.

    Raises SummaryUnavailable when the API refuses or fails.
    """
    import groq  # imported here so the rest of the package works without it

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"], timeout=TIMEOUT_S, max_retries=1)
    try:
        reply = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
    except groq.RateLimitError as error:
        wait = error.response.headers.get("retry-after")
        when = f"in about {wait} seconds" if wait else "in a minute"
        raise SummaryUnavailable(f"The summary service is busy (rate limit reached). Try again {when}.") from error
    except groq.AuthenticationError as error:
        raise SummaryUnavailable("The summary service rejected the API key.") from error
    except (groq.APIConnectionError, groq.APITimeoutError) as error:
        raise SummaryUnavailable("The summary service could not be reached.") from error
    except groq.APIError as error:
        raise SummaryUnavailable("The summary service failed to answer.") from error
    return reply.choices[0].message.content or ""
