import json

from ouroboros.tools import search


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def decode(self):
        return self.read().decode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_tavily_parses_mcp_content_json(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "answer": "Paris",
                            "results": [{"url": "https://example.com", "title": "Example"}],
                        }
                    ),
                }
            ]
        },
    }

    def fake_urlopen(req, timeout=30):
        return _DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = json.loads(search._web_search_tavily("capital of france"))
    assert result["answer"] == "Paris"
    assert result["sources"] == [{"url": "https://example.com", "title": "Example"}]


def test_tavily_falls_back_to_text_when_not_json(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": "plain text answer"}]
        },
    }

    def fake_urlopen(req, timeout=30):
        return _DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = json.loads(search._web_search_tavily("query"))
    assert result["answer"] == "plain text answer"
    assert result["sources"] == []
