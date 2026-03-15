import sys
import types

# supervisor.telegram imports requests at module import time.
sys.modules.setdefault("requests", types.SimpleNamespace())

from supervisor.queue import build_evolution_task_text
from ouroboros.context import _build_compact_readme_for_evolution


def test_build_evolution_task_text_contains_checklist_and_stop_rule():
    text = build_evolution_task_text(7)
    assert text.startswith("EVOLUTION #7")
    assert "Checklist:" in text
    assert "Implement ONE focused improvement" in text
    assert "Stop rule:" in text


def test_compact_readme_prefers_target_sections_and_respects_limit():
    readme = """
# Project

## Architecture
A\nB\nC

## Telegram Bot Commands
/hello

## Optional Configuration (environment variables)
X

## Unrelated
Y
"""
    compact = _build_compact_readme_for_evolution(readme, max_chars=120)
    assert "## Architecture" in compact
    assert "## Telegram Bot Commands" in compact or "## Optional Configuration" in compact
    assert "## Unrelated" not in compact
    assert len(compact) <= 120
