"""Ouroboros — LLM tool loop.

Core loop: send messages to LLM, execute tool calls, repeat until final response.
Extracted from agent.py to keep the agent thin.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros.llm import LLMClient, normalize_reasoning_effort, add_usage
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import compact_tool_history, compact_tool_history_llm
from ouroboros.utils import utc_now_iso, append_jsonl, truncate_for_log, sanitize_tool_args_for_log, sanitize_tool_result_for_log, estimate_tokens

log = logging.getLogger(__name__)

# Pricing from OpenRouter API (2026-02-17). Update periodically via /api/v1/models.
_MODEL_PRICING_STATIC = {
    "anthropic/claude-opus-4.6": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4": (15.0, 1.5, 75.0),
    "anthropic/claude-sonnet-4": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4.6": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4.5": (3.0, 0.30, 15.0),
    "openai/o3": (2.0, 0.50, 8.0),
    "openai/o3-pro": (20.0, 1.0, 80.0),
    "openai/o4-mini": (1.10, 0.275, 4.40),
    "openai/gpt-4.1": (2.0, 0.50, 8.0),
    "openai/gpt-5.2": (1.75, 0.175, 14.0),
    "openai/gpt-5.2-codex": (1.75, 0.175, 14.0),
    "google/gemini-2.5-pro-preview": (1.25, 0.125, 10.0),
    "google/gemini-3-pro-preview": (2.0, 0.20, 12.0),
    "x-ai/grok-3-mini": (0.30, 0.03, 0.50),
    "qwen/qwen3.5-plus-02-15": (0.40, 0.04, 2.40),
}

_pricing_fetched = False
_cached_pricing = None
_pricing_lock = threading.Lock()

def _get_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Lazy-load pricing. On first call, attempts to fetch from OpenRouter API.
    Falls back to static pricing if fetch fails.
    Thread-safe via module-level lock.
    """
    global _pricing_fetched, _cached_pricing

    # Fast path: already fetched (read without lock for performance)
    if _pricing_fetched:
        return _cached_pricing or _MODEL_PRICING_STATIC

    # Slow path: fetch pricing (lock required)
    with _pricing_lock:
        # Double-check after acquiring lock (another thread may have fetched)
        if _pricing_fetched:
            return _cached_pricing or _MODEL_PRICING_STATIC

        _pricing_fetched = True
        _cached_pricing = dict(_MODEL_PRICING_STATIC)

        try:
            from ouroboros.llm import fetch_openrouter_pricing
            _live = fetch_openrouter_pricing()
            if _live and len(_live) > 5:
                _cached_pricing.update(_live)
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("Failed to sync pricing from OpenRouter: %s", e)
            # Reset flag so we retry next time
            _pricing_fetched = False

        return _cached_pricing

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int,
                   cached_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Estimate cost from token counts using known pricing. Returns 0 if model unknown."""
    model_pricing = _get_pricing()
    # Try exact match first
    pricing = model_pricing.get(model)
    if not pricing:
        # Try longest prefix match
        best_match = None
        best_length = 0
        for key, val in model_pricing.items():
            if model and model.startswith(key):
                if len(key) > best_length:
                    best_match = val
                    best_length = len(key)
        pricing = best_match
    if not pricing:
        return 0.0
    input_price, cached_price, output_price = pricing
    # Non-cached input tokens = prompt_tokens - cached_tokens
    regular_input = max(0, prompt_tokens - cached_tokens)
    cost = (
        regular_input * input_price / 1_000_000
        + cached_tokens * cached_price / 1_000_000
        + completion_tokens * output_price / 1_000_000
    )
    return round(cost, 6)

READ_ONLY_PARALLEL_TOOLS = frozenset({
    "repo_read", "repo_list",
    "drive_read", "drive_list",
    "web_search", "codebase_digest", "chat_history",
})

# Stateful browser tools require thread-affinity (Playwright sync uses greenlet)
STATEFUL_BROWSER_TOOLS = frozenset({"browse_page", "browser_action"})

# Tools that modify the repository (create or push commits)
# Used for stagnation detection in evolution tasks.
MODIFYING_TOOLS = frozenset({
    "repo_write_commit",
    "claude_code_edit",
    "repo_commit_push",
})

def _truncate_tool_result(result: Any) -> str:
    """
    Hard-cap tool result string to 15000 characters.
    If truncated, append a note with the original length.
    """
    result_str = str(result)
    if len(result_str) <= 15000:
        return result_str
    original_len = len(result_str)
    return result_str[:15000] + f"\n... (truncated from {original_len} chars)"


def _execute_single_tool(
    tools: ToolRegistry,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    task_id: str = "",
) -> Dict[str, Any]:
    """
    Execute a single tool call and return all needed info.

    Returns dict with: tool_call_id, fn_name, result, is_error, args_for_log, is_code_tool
    """
    fn_name = tc["function"]["name"]
    tool_call_id = tc["id"]
    is_code_tool = fn_name in tools.CODE_TOOLS

    # Parse arguments
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, ValueError) as e:
        result = f"⚠️ TOOL_ARG_ERROR: Could not parse arguments for '{fn_name}': {e}"
        return {
            "tool_call_id": tool_call_id,
            "fn_name": fn_name,
            "result": result,
            "is_error": True,
            "args_for_log": {},
            "is_code_tool": is_code_tool,
        }

    args_for_log = sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {})

    # Execute tool
    tool_ok = True
    try:
        result = tools.execute(fn_name, args)
    except Exception as e:
        tool_ok = False
        result = f"⚠️ TOOL_ERROR ({fn_name}): {type(e).__name__}: {e}"
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "tool_error", "task_id": task_id,
            "tool": fn_name, "args": args_for_log, "error": repr(e),
        })

    # Log tool execution (sanitize secrets from result before persisting)
    append_jsonl(drive_logs / "tools.jsonl", {
        "ts": utc_now_iso(), "tool": fn_name, "task_id": task_id,
        "args": args_for_log,
        "result_preview": sanitize_tool_result_for_log(truncate_for_log(result, 2000)),
    })

    is_error = (not tool_ok) or str(result).startswith("⚠️")

    return {
        "tool_call_id": tool_call_id,
        "fn_name": fn_name,
        "result": result,
        "is_error": is_error,
        "args_for_log": args_for_log,
        "is_code_tool": is_code_tool,
    }


class _StatefulToolExecutor:
    """
    Thread-sticky executor for stateful tools (browser, etc).

    Playwright sync API uses greenlet internally which has strict thread-affinity:
    once a greenlet starts in a thread, all subsequent calls must happen in the same thread.
    This executor ensures browse_page/browser_action always run in the same thread.

    On timeout: we shutdown the executor and create a fresh one to reset state.
    """
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None

    def submit(self, fn, *args, **kwargs):
        """Submit work to the sticky thread. Creates executor on first call."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stateful_tool")
        return self._executor.submit(fn, *args, **kwargs)

    def reset(self):
        """Shutdown current executor and create a fresh one. Used after timeout/error."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def shutdown(self, wait=True, cancel_futures=False):
        """Final cleanup."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None


def _make_timeout_result(
    fn_name: str,
    tool_call_id: str,
    is_code_tool: bool,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    timeout_sec: int,
    task_id: str = "",
    reset_msg: str = "",
) -> Dict[str, Any]:
    """
    Create a timeout error result dictionary and log the timeout event.

    Args:
        reset_msg: Optional additional message (e.g., "Browser state has been reset. ")

    Returns: Dict with tool_call_id, fn_name, result, is_error, args_for_log, is_code_tool
    """
    args_for_log = {}
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
        args_for_log = sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {})
    except Exception:
        pass

    result = (
        f"⚠️ TOOL_TIMEOUT ({fn_name}): exceeded {timeout_sec}s limit. "
        f"The tool is still running in background but control is returned to you. "
        f"{reset_msg}Try a different approach or inform the owner{' about the issue' if not reset_msg else ''}."
    )

    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "tool_timeout",
        "tool": fn_name, "args": args_for_log,
        "timeout_sec": timeout_sec,
    })
    append_jsonl(drive_logs / "tools.jsonl", {
        "ts": utc_now_iso(), "tool": fn_name,
        "args": args_for_log, "result_preview": result,
    })

    return {
        "tool_call_id": tool_call_id,
        "fn_name": fn_name,
        "result": result,
        "is_error": True,
        "args_for_log": args_for_log,
        "is_code_tool": is_code_tool,
    }


def _execute_with_timeout(
    tools: ToolRegistry,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    timeout_sec: int,
    task_id: str = "",
    stateful_executor: Optional[_StatefulToolExecutor] = None,
) -> Dict[str, Any]:
    """
    Execute a tool call with a hard timeout.

    On timeout: returns TOOL_TIMEOUT error so the LLM regains control.
    For stateful tools (browser): resets the sticky executor to recover state.
    For regular tools: the hung worker thread leaks as daemon — watchdog handles recovery.
    """
    fn_name = tc["function"]["name"]
    tool_call_id = tc["id"]
    is_code_tool = fn_name in tools.CODE_TOOLS
    use_stateful = stateful_executor and fn_name in STATEFUL_BROWSER_TOOLS

    # Two distinct paths: stateful (thread-sticky) vs regular (per-call)
    if use_stateful:
        # Stateful executor: submit + wait, reset on timeout
        future = stateful_executor.submit(_execute_single_tool, tools, tc, drive_logs, task_id)
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            stateful_executor.reset()
            reset_msg = "Browser state has been reset. "
            return _make_timeout_result(
                fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                timeout_sec, task_id, reset_msg
            )
    else:
        # Regular executor: explicit lifecycle to avoid shutdown(wait=True) deadlock
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_execute_single_tool, tools, tc, drive_logs, task_id)
            try:
                return future.result(timeout=timeout_sec)
            except TimeoutError:
                return _make_timeout_result(
                    fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                    timeout_sec, task_id, reset_msg=""
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def _handle_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: ToolRegistry,
    drive_logs: pathlib.Path,
    task_id: str,
    stateful_executor: _StatefulToolExecutor,
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """
    Execute tool calls and append results to messages.

    Returns: Number of errors encountered
    """
    # Parallelize only for a strict read-only whitelist; all calls wrapped with timeout.
    can_parallel = (
        len(tool_calls) > 1 and
        all(
            tc.get("function", {}).get("name") in READ_ONLY_PARALLEL_TOOLS
            for tc in tool_calls
        )
    )

    if not can_parallel:
        results = [
            _execute_with_timeout(tools, tc, drive_logs,
                                  tools.get_timeout(tc["function"]["name"]), task_id,
                                  stateful_executor)
            for tc in tool_calls
        ]
    else:
        max_workers = min(len(tool_calls), 8)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_index = {
                executor.submit(
                    _execute_with_timeout, tools, tc, drive_logs,
                    tools.get_timeout(tc["function"]["name"]), task_id,
                    stateful_executor,
                ): idx
                for idx, tc in enumerate(tool_calls)
            }
            results = [None] * len(tool_calls)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # Process results in original order
    return _process_tool_results(results, messages, llm_trace, emit_progress)


def _handle_text_response(
    content: Optional[str],
    llm_trace: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Handle LLM response without tool calls (final response).

    Returns: (final_text, accumulated_usage, llm_trace)
    """
    if content and content.strip():
        llm_trace["assistant_notes"].append(content.strip()[:320])
    return (content or ""), accumulated_usage, llm_trace


def _check_budget_limits(
    budget_remaining_usd: Optional[float],
    accumulated_usage: Dict[str, Any],
    round_idx: int,
    messages: List[Dict[str, Any]],
    llm: LLMClient,
    active_model: str,
    active_effort: str,
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    event_queue: Optional[queue.Queue],
    llm_trace: Dict[str, Any],
    task_type: str = "task",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Check budget limits and handle budget overrun.

    Returns:
        None if budget is OK (continue loop)
        (final_text, accumulated_usage, llm_trace) if budget exceeded (stop loop)
    """
    if budget_remaining_usd is None:
        return None

    task_cost = accumulated_usage.get("cost", 0)
    budget_pct = task_cost / budget_remaining_usd if budget_remaining_usd > 0 else 1.0

    if budget_pct > 0.5:
        # Hard stop — protect the budget
        finish_reason = f"Task spent ${task_cost:.3f} (>50% of remaining budget). Stopping."
        log.warning(finish_reason)
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "budget_abort",
            "task_id": task_id, "task_type": task_type,
            "spent_usd": task_cost, "budget_remaining_usd": budget_remaining_usd,
            "reason": finish_reason,
        })
        # Send alert to owner if chat_id available
        if event_queue is not None:
            try:
                from supervisor.state import load_state
                st = load_state()
                owner_chat = int(st.get("owner_chat_id") or 0)
                if owner_chat:
                    event_queue.put({
                        "type": "send_message", "chat_id": owner_chat,
                        "text": (
                            f"⚠️ Budget abort: task `{task_id}` ({task_type}) stopped after "
                            f"${task_cost:.3f} (>50% of remaining).\n"
                            f"Remaining: ${budget_remaining_usd:.3f}"
                        ),
                        "format": "markdown",
                    }, block=False)
            except Exception:
                pass
        return ("", accumulated_usage, llm_trace)

    return None


def _check_stagnation(
    task_type: str,
    round_idx: int,
    llm_trace: Dict[str, Any],
    drive_logs: pathlib.Path,
    task_id: str,
) -> bool:
    """
    Stagnation guard for evolution tasks.

    Returns True if the task should be aborted due to stagnation.
    """
    if task_type != "evolution":
        return False

    # Number of rounds after which we require at least one modifying tool call
    STAGNATION_LIMIT = 10

    if round_idx < STAGNATION_LIMIT:
        return False

    # Check if any tool call in the entire trace so far is a modifying tool
    tool_calls = llm_trace.get("tool_calls", [])
    for call in tool_calls:
        fn_name = call.get("function", {}).get("name")
        if fn_name in MODIFYING_TOOLS:
            return False  # has modified, not stagnant

    # No modifying tool found within the first STAGNATION_LIMIT rounds
    log.warning(
        f"Evolution task {task_id}: no modifying tool calls after {round_idx} rounds. "
        "Aborting due to stagnation."
    )
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "stagnation_abort",
        "task_id": task_id,
        "rounds_without_modification": round_idx,
        "message": "No code-modifying tool used within stagnation limit.",
    })
    if event_queue is not None:
        try:
            from supervisor.state import load_state
            st = load_state()
            owner_chat = int(st.get("owner_chat_id") or 0)
            if owner_chat:
                event_queue.put({
                    "type": "send_message",
                    "chat_id": owner_chat,
                    "text": (
                        f"🛑 Evolution task `{task_id}` aborted due to stagnation: "
                        f"no code modifications after {round_idx} rounds."
                    ),
                    "format": "markdown",
                }, block=False)
        except Exception:
            pass
    return True


def _process_tool_results(
    results: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """Process tool results, append to messages, update trace. Returns number of errors."""
    errors = 0
    for res in results:
        # Record in trace
        llm_trace["tool_calls"].append({
            "tool": res["fn_name"],
            "result_preview": str(res["result"])[:200],
            "is_error": res["is_error"],
        })
        errors += 1 if res["is_error"] else 0

        # Append to messages
        messages.append({
            "role": "tool",
            "tool_call_id": res["tool_call_id"],
            "content": str(res["result"]),
        })
    return errors


def run_llm_loop(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm: LLMClient,
    drive_logs: pathlib.Path,
    emit_progress: Callable[[str], None],
    incoming_messages: queue.Queue,
    task_type: str,
    task_id: str,
    budget_remaining_usd: Optional[float],
    event_queue: Optional[queue.Queue],
    initial_effort: str = "medium",
    drive_root: str = "",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Main LLM loop: exchange messages until a final response (no tool calls) or limit reached.

    Returns: (final_text, accumulated_usage, llm_trace)
    """
    # Initialize
    accumulated_usage: Dict[str, Any] = {}
    llm_trace: Dict[str, Any] = {"assistant_notes": [], "tool_calls": []}
    round_idx = 0
    max_rounds = int(os.getenv("OUROBOROS_MAX_ROUNDS", "200"))

    # Stagnation guard: last modifying tool round
    last_modifying_round = -1

    while True:
        round_idx += 1

        # Get LLM response (may include tool calls)
        text, usage, tool_calls = llm.chat(
            messages=messages,
            tools=tools.get_schemas(),
            effort=initial_effort if round_idx == 1 else None,
        )
        add_usage(accumulated_usage, usage)

        # Handle text-only final response
        if not tool_calls:
            final_text, accumulated_usage, llm_trace = _handle_text_response(
                text, llm_trace, accumulated_usage
            )
            return final_text, accumulated_usage, llm_trace

        # There are tool calls: execute them
        emit_progress(f"Round {round_idx}: executing {len(tool_calls)} tool call(s)...")
        errors = _handle_tool_calls(
            tool_calls=tool_calls,
            tools=tools,
            drive_logs=drive_logs,
            task_id=task_id,
            stateful_executor=_StatefulToolExecutor(),
            messages=messages,
            llm_trace=llm_trace,
            emit_progress=emit_progress,
        )

        # Check for modifying tools to reset stagnation counter
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name")
            if fn_name in MODIFYING_TOOLS:
                last_modifying_round = round_idx

        # Check stagnation guard for evolution tasks
        if task_type == "evolution":
            # If no modifying tool in the first STAGNATION_LIMIT rounds, abort
            if last_modifying_round == -1 and round_idx >= 10:
                log.warning(
                    f"Stagnation in evolution task {task_id}: "
                    f"no modifying tool after {round_idx} rounds. Aborting."
                )
                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "stagnation_abort",
                    "task_id": task_id,
                    "rounds_without_modification": round_idx,
                })
                # Notify owner
                try:
                    from supervisor.state import load_state
                    st = load_state()
                    owner_chat = int(st.get("owner_chat_id") or 0)
                    if owner_chat and event_queue:
                        event_queue.put({
                            "type": "send_message",
                            "chat_id": owner_chat,
                            "text": (
                                f"🛑 Evolution task `{task_id}` aborted due to stagnation: "
                                f"no code modifications after {round_idx} rounds."
                            ),
                            "format": "markdown",
                        }, block=False)
                except Exception:
                    pass
                # Return a final message indicating abort
                return (
                    f"🛑 Evolution task aborted: no code-modifying tool calls after {round_idx} rounds. "
                    "Evolution requires concrete action. Please restart with a clear commit intent.",
                    accumulated_usage,
                    llm_trace
                )

        # Check budget limits after this round
        if budget_remaining_usd is not None:
            budget_check = _check_budget_limits(
                budget_remaining_usd=budget_remaining_usd,
                accumulated_usage=accumulated_usage,
                round_idx=round_idx,
                messages=messages,
                llm=llm,
                active_model=llm.model,
                active_effort=initial_effort,
                max_retries=3,
                drive_logs=drive_logs,
                task_id=task_id,
                event_queue=event_queue,
                llm_trace=llm_trace,
                task_type=task_type,
            )
            if budget_check is not None:
                final_text, _, _ = budget_check
                return final_text, accumulated_usage, llm_trace

        # Hard round limit
        if round_idx >= max_rounds:
            log.warning(f"Task {task_id} reached max rounds ({max_rounds}). Stopping.")
            append_jsonl(drive_logs / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "max_rounds_hit",
                "task_id": task_id,
                "rounds": round_idx,
            })
            return (
                f"⚠️ Max rounds ({max_rounds}) reached. Stopping.",
                accumulated_usage,
                llm_trace
            )

        # Continue to next round
        # Note: tool results already appended to messages by _handle_tool_calls

    # End while
