import pathlib

from supervisor.events import _handle_task_done


class DummyCtx:
    def __init__(self):
        self._state = {"evolution_consecutive_failures": 0}
        self.saved = []
        self.RUNNING = {}
        self.WORKERS = {}
        self.DRIVE_ROOT = pathlib.Path("/tmp")
        self.appended = []

    def load_state(self):
        return dict(self._state)

    def save_state(self, st):
        self._state = dict(st)
        self.saved.append(dict(st))

    def append_jsonl(self, _path, payload):
        self.appended.append(payload)

    def persist_queue_snapshot(self, reason=""):
        return None


def test_evolution_task_done_with_explicit_success_resets_failures():
    ctx = DummyCtx()
    ctx._state["evolution_consecutive_failures"] = 2

    _handle_task_done(
        {
            "task_id": "abc",
            "task_type": "evolution",
            "had_error": False,
            "empty_response": False,
            "cost_usd": 0.0,
            "total_rounds": 0,
        },
        ctx,
    )

    assert ctx._state["evolution_consecutive_failures"] == 0


def test_evolution_task_done_with_error_increments_failures():
    ctx = DummyCtx()

    _handle_task_done(
        {
            "task_id": "abc",
            "task_type": "evolution",
            "had_error": True,
            "empty_response": False,
            "response_len": 50,
            "cost_usd": 0.5,
            "total_rounds": 2,
        },
        ctx,
    )

    assert ctx._state["evolution_consecutive_failures"] == 1
    assert any(e.get("type") == "evolution_task_failure_tracked" for e in ctx.appended)


def test_evolution_task_done_legacy_heuristic_still_applies_without_flags():
    ctx = DummyCtx()

    _handle_task_done(
        {
            "task_id": "abc",
            "task_type": "evolution",
            "cost_usd": 0.01,
            "total_rounds": 1,
        },
        ctx,
    )

    assert ctx._state["evolution_consecutive_failures"] == 1
