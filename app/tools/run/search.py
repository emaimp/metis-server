"""SearXNG web search tool.

Queries a local SearXNG instance (Docker) through its JSON API and formats
the results as context for the model. The answer itself is always produced
by the normal chat flow (ask_chat / ask_chat_stream): this tool only brings
fresh web evidence.
"""

import requests

from app.core.settings import (
    LANGUAGE_CODE,
    MAX_SEARCH_RESULTS,
    SEARCH_SNIPPET_CHARS,
    SEARXNG_BASE_URL,
    SEARXNG_TIMEOUT,
)


def search_web(query: str, categories: str = 'general') -> list[dict]:
    """Run a web search against the configured SearXNG instance.

    Args:
        query: The search query (the user's question without the tool mention).
        categories: SearXNG category ('general', 'news', 'science', ...).

    Returns:
        A list (at most MAX_SEARCH_RESULTS) of result dicts with at least
        'title', 'url' and 'content'. An empty list means "no results" and is
        NOT an error: the model simply answers from its own knowledge.

    Raises:
        ConnectionError: SearXNG is unreachable, misconfigured or returned an
            HTTP error (mapped to 502 by the router, like llama.cpp failures).
        RuntimeError: The response body is not valid JSON.
    """
    if not SEARXNG_BASE_URL:
        raise ConnectionError('SEARXNG_BASE_URL is not configured')
    try:
        resp = requests.get(
            f'{SEARXNG_BASE_URL}/search',
            params={
                'q': query,
                'format': 'json',
                'categories': categories,
                'language': LANGUAGE_CODE,
                'safesearch': 1,
            },
            timeout=SEARXNG_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f'Cannot connect to SearXNG: {e}') from e
    except ValueError as e:
        raise RuntimeError(f'Unexpected response from SearXNG: {e}') from e

    results = payload.get('results')
    if not isinstance(results, list):
        return []
    # Keep only well-formed entries: an item without URL is not usable as a source.
    return [
        r for r in results[:MAX_SEARCH_RESULTS * 2]
        if isinstance(r, dict) and str(r.get('url') or '').strip()
    ][:MAX_SEARCH_RESULTS]


def format_search_context(results: list[dict]) -> str:
    """Format SearXNG results as a numbered Título/URL/Contenido context block."""
    if not results:
        return 'Web search returned no results.'
    blocks = []
    for i, r in enumerate(results, 1):
        title = str(r.get('title') or '').strip() or '(no title)'
        url = str(r.get('url') or '').strip()
        content = str(r.get('content') or '').strip()[:SEARCH_SNIPPET_CHARS]
        blocks.append(f'[{i}] {title}\nURL: {url}\n{content}')
    return '\n\n'.join(blocks)
