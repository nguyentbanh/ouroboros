import datetime

from supervisor.queue import QueueStore


def _make_store():
    return QueueStore(pending=[], running={}, seq_counter_ref={"value": 0}, lock=None)


def test_queue_store_enqueue_assigns_defaults_and_orders_by_priority():
    store = _make_store()

    low = store.enqueue({"id": "a", "type": "evolution", "chat_id": 1, "text": "x"})
    high = store.enqueue({"id": "b", "type": "task", "chat_id": 1, "text": "y"})

    assert low["_attempt"] == 1
    assert "queued_at" in low
    # task priority (0) should be before evolution (1)
    assert [t["id"] for t in store.pending] == ["b", "a"]
    # sequence should keep incrementing regardless of sort position
    assert high["_queue_seq"] == 2


def test_queue_store_restore_skips_invalid_rows():
    store = _make_store()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    snapshot = {
        "ts": now,
        "pending": [
            {"task": {"id": "ok", "type": "task", "chat_id": 1, "text": "run"}},
            {"task": {"id": "missing_chat", "type": "task"}},
            {"foo": "bar"},
        ],
    }

    restored = store.restore(snapshot, max_age_sec=60)

    assert restored == 1
    assert len(store.pending) == 1
    assert store.pending[0]["id"] == "ok"


def test_queue_store_cancel_only_removes_pending_by_default():
    store = _make_store()
    store.pending.append({"id": "p1"})
    store.running["r1"] = {"task": {"id": "r1", "type": "task"}}

    assert store.cancel("p1") == "pending"
    assert "r1" in store.running
    assert store.cancel("r1") is None
    assert store.cancel("r1", include_running=True) == "running"
    assert "r1" not in store.running
