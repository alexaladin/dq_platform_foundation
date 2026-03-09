from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AISuggestPatchResponse:
    """
    Patch-mode response: AI suggests only additions, never full ruleset.
    """

    rules_to_add: list[dict[str, Any]]
    rationale: str
    raw: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    latency_ms: int | None = None
