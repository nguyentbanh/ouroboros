import queue
from pathlib import Path

from ouroboros.loop import run_llm_loop
from ouroboros.tools.registry import ToolContext


class _FakeTools:
    CODE_TOOLS = frozenset()

    def __init__(self, ctx):
        self._ctx = ctx

    def schemas(self):
        return []


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def default_model(self):
        return "test/model"

    def chat(self, messages, model, tools=None, reasoning_effort="medium"):
        self.calls.append(messages)
        return {"content": "done", "tool_calls": []}, {"prompt_tokens": 1, "completion_tokens": 1}


def _make_messages(tool_result_text: str):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{"id": "t1", "function": {"name": "repo_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": tool_result_text},
    ]


def _make_messages_three_rounds(old_tool_result_text: str):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "round1",
            "tool_calls": [{"id": "t1", "function": {"name": "repo_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": old_tool_result_text},
        {
            "role": "assistant",
            "content": "round2",
            "tool_calls": [{"id": "t2", "function": {"name": "repo_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t2", "content": "ok"},
        {
            "role": "assistant",
            "content": "round3",
            "tool_calls": [{"id": "t3", "function": {"name": "repo_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t3", "content": "ok"},
    ]


def test_pending_compaction_is_applied_before_chat(monkeypatch, tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx._pending_compaction = 2
    tools = _FakeTools(ctx)
    llm = _FakeLLM()

    def _fake_compact(messages, keep_recent=6):
        compacted = list(messages)
        compacted[-1] = {**compacted[-1], "content": "COMPACTED"}
        return compacted

    monkeypatch.setattr("ouroboros.loop.compact_tool_history_llm", _fake_compact)

    final_text, usage, _trace = run_llm_loop(
        messages=_make_messages("X" * 2000),
        tools=tools,
        llm=llm,
        drive_logs=Path(tmp_path),
        emit_progress=lambda _m: None,
        incoming_messages=queue.Queue(),
        task_type="user",
        task_id="t",
        budget_remaining_usd=None,
        event_queue=None,
    )

    assert final_text == "done"
    assert usage["prompt_tokens"] == 1
    assert ctx._pending_compaction is None
    assert llm.calls[0][-1]["content"] == "COMPACTED"


def test_auto_compaction_trims_large_tool_results(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_AUTO_COMPACT_TOKENS", "10")
    monkeypatch.setenv("OUROBOROS_AUTO_COMPACT_KEEP_RECENT", "2")

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    tools = _FakeTools(ctx)
    llm = _FakeLLM()

    run_llm_loop(
        messages=_make_messages_three_rounds("A" * 5000),
        tools=tools,
        llm=llm,
        drive_logs=Path(tmp_path),
        emit_progress=lambda _m: None,
        incoming_messages=queue.Queue(),
        task_type="user",
        task_id="t",
        budget_remaining_usd=None,
        event_queue=None,
    )

    # Auto compaction should shorten the tool message before first LLM call.
    assert len(llm.calls[0][3]["content"]) < 5000
