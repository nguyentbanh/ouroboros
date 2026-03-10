import datetime
import threading

from supervisor.queue import QueueStore


def _make_store():
    return QueueStore(
        pending=[], running={}, seq_counter_ref={"value": 0}, lock=threading.RLock()
    )


def test_queue_store_enqueue_assigns_defaults_and_orders_by_priority():
    store = _make_store()

    low = store.enqueue({"id": "a", "type": "evolution", "chat_id": 1, "text": "x"})
    high = store.enqueue({"id": "b", "type": "task", "chat_id": 1, "text": "y"})

    assert low["_attempt"] == 1
    assert "queued_at" in low
    assert [t["id"] for t in store.pending] == ["b", "a"]
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


def test_queue_store_thread_safe_enqueue_sequence_unique():
    store = _make_store()

    def worker(start: int):
        for i in range(50):
            store.enqueue(
                {"id": f"{start+i}", "type": "task", "chat_id": 1, "text": "x"}
            )

    threads = [threading.Thread(target=worker, args=(idx * 1000,)) for idx in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    seqs = [int(t["_queue_seq"]) for t in store.pending]
    assert len(seqs) == 200
    assert len(set(seqs)) == 200
    assert min(seqs) == 1
    assert max(seqs) == 200
