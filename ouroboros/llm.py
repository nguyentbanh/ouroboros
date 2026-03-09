"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter or OpenAI Codex).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """OpenRouter API wrapper (or OpenAI Codex via OAuth). All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Check for Codex OAuth mode
        use_codex_oauth = os.environ.get("OUROBOROS_USE_CODEX_OAUTH", "").lower() == "true"
        codex_token = os.environ.get("CODEX_OAUTH_TOKEN", "")

        if use_codex_oauth and codex_token:
            # Codex OAuth mode: use OpenAI API with Codex token
            self._api_key = codex_token
            self._base_url = "https://api.openai.com/v1"
            self._using_codex_oauth = True
            log.info("LLMClient initialized in Codex OAuth mode (OpenAI API)")
        else:
            # Default: OpenRouter
            self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            self._base_url = base_url or "https://openrouter.ai/api/v1"
            self._using_codex_oauth = False
            log.info("LLMClient initialized in OpenRouter mode")

        self._client = None

        # Fallback configuration (for OpenRouter rate-limit resilience)
        self._primary_model = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        self._fallback_chain = []
        self._fallback_index = 0
        self._max_retries = 3
        self._base_backoff = 1.0
        if not self._using_codex_oauth:
            fb_str = os.environ.get("OUROBOROS_MODEL_FALLBACKS", "")
            if fb_str:
                self._fallback_chain = [m.strip() for m in fb_str.split(",") if m.strip()]
            try:
                self._max_retries = int(os.environ.get("OUROBOROS_FALLBACK_MAX_RETRIES", "3"))
            except ValueError:
                pass
            try:
                self._base_backoff = float(os.environ.get("OUROBOROS_FALLBACK_BASE_BACKOFF", "1.0"))
            except ValueError:
                pass

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        if self._using_codex_oauth:
            # Cost estimation for Codex: not available via OpenAI API; skip
            return None
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat_completion(self, *args, **kwargs):
        return self.chat(*args, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call with optional fallback chain for rate limits. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }

        # Pin Anthropic models to Anthropic provider for prompt caching (OpenRouter only)
        if not self._using_codex_oauth and model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if tools:
            # Add cache_control to last tool for Anthropic prompt caching (OpenRouter only)
            # This caches all tool schemas (they never change between calls)
            if not self._using_codex_oauth:
                tools_with_cache = [t for t in tools]  # shallow copy
                if tools_with_cache:
                    last_tool = {**tools_with_cache[-1]}  # copy last tool
                    last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                    tools_with_cache[-1] = last_tool
                kwargs["tools"] = tools_with_cache
            else:
                kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Fallback logic for rate limits (only for primary model in OpenRouter mode)
        use_fallback = (model == self._primary_model) and bool(self._fallback_chain) and not self._using_codex_oauth
        if use_fallback:
            models_to_try = [self._primary_model] + self._fallback_chain
        else:
            models_to_try = [model]

        last_exception = None
        for attempt_idx, attempt_model in enumerate(models_to_try):
            try:
                # Prepare kwargs without model for this attempt
                attempt_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
                resp = client.chat.completions.create(model=attempt_model, **attempt_kwargs)
                # On success, update fallback index if using fallback
                if use_fallback:
                    self._fallback_index = attempt_idx
                break
            except Exception as e:
                last_exception = e
                # Determine if error is retryable
                retryable = False
                status = getattr(e, 'status_code', None)
                if status in (429, 500, 502, 503, 504):
                    retryable = True
                elif isinstance(e, (RateLimitError, APIError, APIConnectionError, APITimeoutError)):
                    retryable = True

                # If not retryable or last model, re-raise
                if not retryable or attempt_idx == len(models_to_try) - 1:
                    raise

                # Log fallback event and backoff
                log.warning(
                    f"Model '{attempt_model}' failed with {type(e).__name__} (status={status}). "
                    f"Falling back to next model (attempt {attempt_idx+1}/{len(models_to_try)})..."
                )
                backoff_seconds = self._base_backoff * (2 ** attempt_idx)
                time.sleep(backoff_seconds)
                continue

        if last_exception is not None and 'resp' not in locals():
            raise last_exception

        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage
        if not usage.get("cost"):
            if self._using_codex_oauth:
                # For Codex OAuth, estimate cost using static pricing table in loop.py
                # The loop will call _estimate_cost; we just need token counts
                pass
            else:
                gen_id = resp_dict.get("id") or ""
                if gen_id:
                    cost = self._fetch_generation_cost(gen_id)
                    if cost is not None:
                        usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models