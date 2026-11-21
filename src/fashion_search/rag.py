"""Summaries of a result list written by an LLM (retrieval-augmented generation).

The search engine retrieves the products and an LLM, reached through the Groq
API as in the course template, writes a short recommendation from them. Each
way of building the prompt is a *variant*, so the baseline and the improved
versions can be compared on the same queries (see notebooks/part_4.ipynb).

Set GROQ_API_KEY (and optionally GROQ_MODEL) in .env. Without a key,
`summarize` returns None, so callers can simply leave the summary out.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fashion_search.search import Result

DEFAULT_MODEL = "llama-3.1-8b-instant"  # the course template's model

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
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    prompt = VARIANTS[variant](query, results)
    return Summary(text=complete(prompt, model), variant=variant, model=model)


def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send one user message to the Groq API and return the reply's text."""
    from groq import Groq  # imported here so the rest of the package works without it

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    reply = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return reply.choices[0].message.content or ""
