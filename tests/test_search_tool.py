"""Unit tests for the SearXNG web search tool (app/tools/run/search.py)."""

import pathlib
import sys

# Ensure the project root (containing the 'app' package) is importable
# when pytest is run from anywhere.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import requests

from app.tools.run.search import format_search_context, search_web


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error')

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_search_web_returns_trimmed_results(monkeypatch):
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured['url'] = url
        captured['params'] = params
        captured['timeout'] = timeout
        return _FakeResponse({
            'results': [
                {'title': f'T{i}', 'url': f'https://e.com/{i}', 'content': 'c' * 10}
                for i in range(8)
            ],
        })

    monkeypatch.setattr(requests, 'get', _fake_get)
    results = search_web('llama.cpp news')

    assert len(results) == 5  # MAX_SEARCH_RESULTS cap
    assert results[0]['title'] == 'T0'
    assert captured['url'].endswith('/search')
    assert captured['params']['q'] == 'llama.cpp news'
    assert captured['params']['format'] == 'json'
    assert captured['params']['categories'] == 'general'


def test_search_web_skips_malformed_entries(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: _FakeResponse({
        'results': [
            {'title': 'ok', 'url': 'https://e.com', 'content': 'fine'},
            'not-a-dict',
            {'title': 'no url'},
            {'url': '   '},
        ],
    }))
    results = search_web('q')
    assert [r['title'] for r in results] == ['ok']


def test_search_web_no_results_is_not_an_error(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: _FakeResponse({'results': []}))
    assert search_web('q') == []


def test_search_web_missing_results_key(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: _FakeResponse({}))
    assert search_web('q') == []


def test_search_web_connection_error(monkeypatch):
    def _boom(url, params=None, timeout=None):
        raise requests.ConnectionError('refused')

    monkeypatch.setattr(requests, 'get', _boom)
    with pytest.raises(ConnectionError):
        search_web('q')


def test_search_web_http_error_maps_to_connection_error(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: _FakeResponse({}, status_code=403))
    with pytest.raises(ConnectionError):
        search_web('q')


def test_search_web_invalid_json(monkeypatch):
    monkeypatch.setattr(
        requests, 'get', lambda url, **kw: _FakeResponse(ValueError('bad json')),
    )
    with pytest.raises(RuntimeError):
        search_web('q')


def test_format_search_context_results():
    context = format_search_context([
        {'title': 'llama.cpp release', 'url': 'https://e.com/a', 'content': 'Novedades'},
        {'title': 'otra', 'url': 'https://e.com/b', 'content': 'x' * 500},
    ])
    assert '[1] llama.cpp release' in context
    assert 'URL: https://e.com/a' in context
    assert '[2] otra' in context
    assert len(context.split('[2]')[1]) <= 350  # snippet truncated


def test_format_search_context_empty():
    assert format_search_context([]) == 'Web search returned no results.'
