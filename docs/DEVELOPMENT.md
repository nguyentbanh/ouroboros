# Development Guide — Ouroboros

This document summarizes conventions, tooling, and invariants for developing Ouroboros.

## Quick start

```bash
# Run tests
make test

# Generate self-awareness dashboard
make self-awareness

# Run health check (complexity metrics)
make health
```

## Branches

- `main` — upstream stable, read-only for the agent.
- `ouroboros` — agent's working branch. All development commits here.
- `ouroboros-stable` — fallback branch. Promote when stable.

## Versioning and Release Invariant (Bible P7)

Every release must keep three sources in sync:

- `VERSION` file
- `README.md` version line
- Git tag (annotated) `v{VERSION}`

**Automated guard:** A pre-commit hook (versioned as `hooks/pre-commit`) checks the invariant before every commit on `ouroboros`. Install it once:

```bash
chmod +x hooks/pre-commit
cp hooks/pre-commit .git/hooks/pre-commit
```

The hook fails fast with instructions if `VERSION` and `README.md` differ.

To skip the hook (emergency only):

```bash
git commit --no-verify -m "urgent fix"
```

**Release sequence:**

1. Update `VERSION` (semver) and `README.md` changelog.
2. Commit: `v{VERSION}: brief description`.
3. Create annotated tag and push:

```bash
git tag -a v{VERSION} -m "v{VERSION}: description"
git push origin v{VERSION}
```

4. Optionally create GitHub Release (major/minor).
5. `promote_to_stable` when confident.

**Invariant:** `VERSION == README.md == latest git tag`. Discrepancy is a critical bug.

## Code Quality

- Minimalism (P5): modules ≤ 1000 lines, functions ≤ 150 lines, ≤ 8 parameters.
- Smoke tests (`tests/test_smoke.py`) run on every push via `make test`.
- Complexity metrics: `make health`.

## Tooling Conventions

- Code edits: prefer `claude_code_edit` → `repo_commit_push`.
- Small edits: `repo_write_commit`.
- Deep review: `request_review` for architecture/security changes.
- Multi-model review: `multi_model_review` with diverse models.

## Knowledge Base

Record recipes, pitfalls, and API quirks in the Drive knowledge base:

```python
knowledge_write(topic='topic-name', content='...', mode='append')
```

## Background Consciousness

Start/stop with `/bg start` / `/bg stop`. The agent maintains its own thinking budget (default 10% of total).

## Evolution Mode

Enable with `/evolve`. Each cycle must produce a commit and version bump. Analyze → implement → test → commit → restart. Pause if consecutive failures occur.

## Invariants to Monitor

- VERSION sync (pre-commit hook)
- Budget drift ≤ 20%
- No duplicate message processing
- Identity freshness (update after significant dialogue)

When in doubt, consult `BIBLE.md` first.
