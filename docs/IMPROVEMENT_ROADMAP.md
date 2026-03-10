# Improvement Roadmap: Simplify Ouroboros + Add High-Impact Features

This roadmap focuses on two goals:

1. **Reduce complexity** in core loops and operational code.
2. **Add features** that improve reliability, observability, and autonomous evolution quality.

---

## 1) Simplify the Current Codebase

### A. Extract queue state into a small `QueueStore` object

**Current pain:** `supervisor/queue.py` uses module-level globals (`PENDING`, `RUNNING`, lock refs), making concurrency assumptions spread across many functions.

**Simplification:**
- Create a `QueueStore` dataclass/class with:
  - `pending`, `running`, `seq_counter`, `lock`
  - methods: `enqueue()`, `dequeue()`, `snapshot()`, `restore()`, `cancel()`
- Keep public wrappers for backward compatibility while gradually migrating call sites.

**Benefits:**
- Fewer implicit invariants.
- Easier unit testing and reasoning about locking.
- Lower risk of race conditions.

---

### B. Normalize model fallback config in one place

**Current pain:** defaults and fallback logic are split between docs and `LLMClient` runtime behavior.

**Simplification:**
- Add `resolve_model_chain(env: Mapping[str, str]) -> list[str]` in `ouroboros/llm.py`.
- Single precedence order:
  1. `OUROBOROS_MODEL_FALLBACK_LIST`
  2. `OUROBOROS_MODEL_FALLBACKS` (legacy)
  3. canonical hardcoded default
- Reuse this function both in constructor and tests.

**Benefits:**
- Predictable behavior.
- Docs become a direct reflection of code.

---

### C. Introduce typed internal event records

**Current pain:** many dict-shaped records (`task`, `running meta`, logs) with ad-hoc keys.

**Simplification:**
- Add lightweight dataclasses (`TaskRecord`, `RunningTaskMeta`, `QueueSnapshot`).
- Convert to/from dict at boundaries only (JSON, API, persistence).

**Benefits:**
- Fewer key-typo bugs.
- Better IDE assistance and clearer contracts.

---

### D. Reduce "god function" pressure in tool loop

**Current pain:** `ouroboros/loop.py` contains many responsibilities (pricing, execution policy, parallelization, truncation, stagnation checks).

**Simplification:**
- Split into modules:
  - `loop_pricing.py`
  - `loop_execution.py`
  - `loop_policies.py`
  - `loop_logging.py`
- Keep `run_tool_loop()` as orchestrator only.

**Benefits:**
- Smaller modules aligned with principle of minimalism.
- Easier independent testing.

---

## 2) Reliability Upgrades (short-term)

### A. Queue concurrency test suite (priority)
- Stress tests with concurrent enqueue/check/cancel.
- Assert deterministic ordering and no exceptions.
- Simulate stale heartbeat + timeout enforcement paths.

### B. Snapshot integrity checks
- Add schema validation before restore.
- Store snapshot format version (`snapshot_version`).
- Graceful migration path for old snapshots.

### C. Failure taxonomy for evolution cycles
- Track failure classes (`llm_empty`, `tool_error`, `timeout`, `commit_missing`, etc.).
- Use this taxonomy to adapt retries and model switching.

---

## 3) New Features to Add Next

### 1. Evolution Planner (high impact)
Before each `/evolve` cycle, generate a compact plan:
- one objective
- expected files
- test/check checklist
- rollback strategy

Store as JSON in logs, and compare planned vs actual outcome.

**Why:** increases coherence and prevents random micro-edits.

---

### 2. Automatic Post-Change Verification Matrix
After any repo-modifying tool call:
- run targeted tests
- run smoke checks
- run lint/format checks
- attach pass/fail summary to task result

**Why:** reduces broken self-modifications.

---

### 3. Tool Safety Profiles
Define per-tool risk levels:
- `read_only`
- `repo_mutation`
- `network_side_effect`
- `process_control`

Require extra confirmation logic internally for high-risk chains (e.g., force intermediate self-review step).

**Why:** safer autonomy without removing power.

---

### 4. Memory Quality Scoring
Add a score for memory entries based on:
- recency
- reuse frequency
- source reliability
- contradiction flags

Use score to prioritize what enters prompt context.

**Why:** cheaper prompts + better continuity.

---

### 5. Budget-Aware Reasoning Effort
Dynamically lower/raise reasoning effort and model choice based on:
- remaining budget
- task criticality
- prior attempt failures

**Why:** preserves budget while maintaining quality on important tasks.

---

### 6. Self-Review Diff Narrator
For each commit, auto-generate:
- what changed
- why
- expected user-visible impact
- risk notes
- revert instructions

**Why:** improves traceability and trust in autonomous changes.

---

## 4) Suggested Execution Order (pragmatic)

1. Queue lock/state refactor + tests
2. Fallback config unification + tests + docs sync
3. Post-change verification matrix
4. Evolution planner
5. Memory quality scoring
6. Tool safety profiles

---

## 5) Definition of Done for "simplification"

A change is considered simplification only if it meets all:
- Fewer mutable global touchpoints
- Lower branch complexity in modified functions
- Same or better test coverage for touched logic
- No increase in required operator configuration

---

## 6) Anti-Goals (avoid)

- Adding more tools before stabilizing queue and loop reliability.
- Expanding prompt size without memory quality controls.
- Introducing framework-heavy abstractions that conflict with minimalism.

---

## 7) Minimal KPI Set

Track weekly:
- evolution success rate (commit produced + checks passed)
- rollback rate
- mean task completion latency
- budget per successful evolution cycle
- timeout-induced worker restarts

If these improve while LOC/complexity stays flat, the roadmap is working.
