"""Web search tool with Tavily MCP and OpenAI fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


def _web_search_openai(query: str) -> str:
    """Search using OpenAI Responses API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "OPENAI_API_KEY not set; web_search unavailable."})
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=os.environ.get("OUROBOROS_WEBSEARCH_MODEL", "gpt-5"),
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=query,
        )
        d = resp.model_dump()
        text = ""
        sources = []
        for item in d.get("output", []) or []:
            if item.get("type") == "message":
                for block in item.get("content", []) or []:
                    if block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")
                        # OpenAI Responses may include source annotations in the message
                        for annotation in block.get("annotations", []):
                            if annotation.get("type") == "url_citation":
                                sources.append({
                                    "url": annotation.get("url", ""),
                                    "title": annotation.get("title", ""),
                                })
        return json.dumps({"answer": text or "(no answer)", "sources": sources}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def _web_search_tavily(query: str) -> str:
    """Search using Tavily remote MCP server."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY not set; Tavily search unavailable."})

    url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"

    # Build JSON-RPC request
    request = {
        "method": "tools/call",
        "params": {
            "name": os.environ.get("TAVILY_TOOL_NAME", "tavily-search"),
            "arguments": {
                "query": query,
                "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", "basic"),
                "max_results": int(os.environ.get("TAVILY_MAX_RESULTS", "10")),
                "include_images": os.environ.get("TAVILY_INCLUDE_IMAGES", "false").lower() == "true",
            }
        },
        "jsonrpc": "2.0",
        "id": 1
    }

    try:
        import urllib.request
        import urllib.error

        req_data = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                # Accept both JSON and event-stream per MCP spec
                "Accept": "application/json, text/event-stream",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Read the full response; MCP servers often return text/event-stream
            # with one or more SSE events containing JSON-RPC messages.
            # For simple queries, the server may return a single JSON directly.
            raw = resp.read().decode("utf-8", errors="ignore")
            content_type = resp.headers.get("Content-Type", "")

        # If it's a single JSON response (non-streaming), parse directly.
        if "application/json" in content_type:
            result = json.loads(raw)
        else:
            # Assume SSE streaming: parse all "data: {...}" lines
            result = None
            for line in raw.splitlines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                        # The final result is typically in a JSON-RPC response with result field
                        if "result" in chunk:
                            result = chunk
                            break
                        # Errors may appear as a single chunk
                        if "error" in chunk:
                            return json.dumps({"error": f"Tavily MCP error: {chunk['error']}"}, ensure_ascii=False)
                    except json.JSONDecodeError:
                        continue
            if result is None:
                return json.dumps({"error": "No valid JSON-RPC response from Tavily MCP"}, ensure_ascii=False)

        if "error" in result:
            return json.dumps({"error": f"Tavily MCP error: {result['error']}"}, ensure_ascii=False)

        # Extract result payload
        payload = result.get("result", {}) or {}

        extracted: Dict[str, Any] = {}
        blocks = payload.get("content", []) or []
        text_fragments: List[str] = []
        for block in blocks:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_fragments.append(block["text"])

        for txt in text_fragments:
            try:
                candidate = json.loads(txt)
            except Exception:
                continue
            if isinstance(candidate, dict):
                extracted = candidate
                break

        if not extracted:
            extracted = {"answer": "\n\n".join(text_fragments)}

        answer = (
            extracted.get("answer", "")
            or extracted.get("text", "")
            or payload.get("answer", "")
            or payload.get("text", "")
        )

        sources = []
        source_items = (
            extracted.get("sources")
            or extracted.get("results")
            or payload.get("sources")
            or []
        )
        for src in source_items:
            if not isinstance(src, dict):
                continue
            sources.append({
                "url": src.get("url", ""),
                "title": src.get("title", "") or src.get("name", ""),
            })

        return json.dumps({
            "answer": answer or "(no answer)",
            "sources": sources
        }, ensure_ascii=False, indent=2)

    except urllib.error.HTTPError as e:
        return json.dumps({"error": f"Tavily HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def _web_search(ctx: ToolContext, query: str) -> str:
    """Route search to Tavily (if configured) or OpenAI."""
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        return _web_search_tavily(query)
    else:
        return _web_search_openai(query)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": (
                "Search the web. Uses Tavily remote MCP server when TAVILY_API_KEY is set; "
                "otherwise falls back to OpenAI Responses API. Returns JSON with answer + sources."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]