from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchDecision:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]  # each rejected rule should include a "reject_reason"


def _freeze(v: Any) -> Any:
    """
    Convert nested dict/list structures into hashable equivalents.
    - dict -> tuple of (key, frozen(value)) sorted by key
    - list/tuple/set -> tuple of frozen items
    - other -> unchanged
    """
    if isinstance(v, dict):
        return tuple((k, _freeze(vv)) for k, vv in sorted(v.items(), key=lambda x: str(x[0])))
    if isinstance(v, (list, tuple, set)):
        return tuple(_freeze(x) for x in v)
    return v


def _signature(rule: dict[str, Any]) -> tuple:
    rtype = rule.get("rule_type")
    cols = tuple(rule.get("columns") or [])
    params = rule.get("params") or {}
    params_frozen = _freeze(params)
    return (rtype, cols, params_frozen)


def _reject(rule: dict[str, Any], reason: str) -> dict[str, Any]:
    out = dict(rule)
    out["reject_reason"] = reason
    return out


def validate_and_filter_ai_rules(
    *,
    ai_rules: list[dict[str, Any]],
    allowed_rule_types: list[str],
    dataset_columns: set[str],
    existing_rules: list[dict[str, Any]],
    max_rules_to_add: int = 15,
    min_ai_confidence: float | None = None,
) -> PatchDecision:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    existing_sigs = {_signature(r) for r in existing_rules if isinstance(r, dict)}

    for r in ai_rules:
        if not isinstance(r, dict):
            rejected.append(_reject({"raw": r}, "rule is not a dict"))
            continue

        rtype = r.get("rule_type")
        if rtype not in allowed_rule_types:
            rejected.append(_reject(r, f"type not allowed: {rtype}"))
            continue

        # Confidence filter (optional)
        if min_ai_confidence is not None:
            conf = r.get("confidence")
            try:
                conf_f = float(conf) if conf is not None else None
            except Exception:
                conf_f = None
            if conf_f is None or conf_f < min_ai_confidence:
                rejected.append(_reject(r, f"confidence below threshold: {conf}"))
                continue

        # Column checks for column-based rules
        col_rules = {"completeness", "uniqueness", "domain", "range", "date_not_in_future"}
        if rtype in col_rules:
            cols = r.get("columns")
            if not isinstance(cols, list) or len(cols) == 0:
                rejected.append(_reject(r, "missing/invalid columns"))
                continue
            if any((not isinstance(c, str) or c.strip() == "") for c in cols):
                rejected.append(_reject(r, "columns must be non-empty strings"))
                continue
            if any(c not in dataset_columns for c in cols):
                missing = [c for c in cols if c not in dataset_columns]
                rejected.append(_reject(r, f"columns not in dataset: {missing}"))
                continue

        # Params validation for some types
        params = r.get("params") or {}
        if params is not None and not isinstance(params, dict):
            rejected.append(_reject(r, "params must be dict"))
            continue

        if rtype == "domain":
            # allow either allowed_values OR max_distinct (for mock/demo), but enforce at least one
            if "allowed_values" not in params and "max_distinct" not in params:
                rejected.append(_reject(r, "domain.params must include allowed_values (preferred)"))
                continue

        if rtype == "range":
            if "min" not in params and "max" not in params:
                rejected.append(_reject(r, "range.params must include min and/or max"))
                continue

        sig = _signature(r)
        if sig in existing_sigs:
            rejected.append(_reject(r, "duplicate rule (signature already exists)"))
            continue

        accepted.append(r)
        existing_sigs.add(sig)

        if len(accepted) >= max_rules_to_add:
            break

    return PatchDecision(accepted=accepted, rejected=rejected)
