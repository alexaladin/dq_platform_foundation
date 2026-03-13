from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchDecision:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


# ---- helpers: make hashable ----
def _freeze(v: Any) -> Any:
    if isinstance(v, dict):
        items = []
        for k, vv in v.items():
            if k == "allowed_values" and isinstance(vv, (list, tuple, set)):
                frozen_vals = tuple(_freeze(x) for x in vv)
                # order-insensitive for allowed_values
                try:
                    frozen_vals = tuple(sorted(frozen_vals))
                except Exception:
                    pass
                items.append((k, frozen_vals))
            else:
                items.append((k, _freeze(vv)))
        return tuple(sorted(items, key=lambda x: str(x[0])))

    if isinstance(v, (list, tuple, set)):
        return tuple(_freeze(x) for x in v)

    return v


def _normalize_sql_refs(rule: dict[str, Any]) -> tuple:
    """
    Normalize sql_ref list for dedup signature in etl_validation rules.
    Returns a sorted, frozen tuple of sql_ref items.
    """
    exp = rule.get("expectation") or {}
    sql_refs = exp.get("sql_ref") or []
    if not isinstance(sql_refs, list):
        return ()
    # Normalize each ref: sort keys, strip whitespace from values
    normalized = []
    for ref in sql_refs:
        if not isinstance(ref, dict):
            continue
        if "file" in ref:
            normalized.append(("file", str(ref["file"]).strip()))
        elif "inline_sql" in ref:
            # Normalize whitespace for inline SQL dedup
            normalized.append(("inline_sql", " ".join(str(ref["inline_sql"]).split())))
    # Sort for order-insensitive dedup
    try:
        normalized.sort()
    except Exception:
        pass
    return tuple(normalized)


def _normalize_cols(rule_type: str | None, rule: dict[str, Any]) -> list[str]:
    """
    Normalize where columns live:
      - canonical patch: rule["column"] (str)
      - legacy patch: rule["columns"] (list)
      - canonical: rule["expectation"]["column"] (str)
      - schema: expectation.required_columns (list)
      - freshness: expectation.ts_column (str)
    """
    # canonical patch format
    col = rule.get("column")
    if isinstance(col, str) and col.strip():
        return [col]

    # legacy patch format
    cols = rule.get("columns")
    if isinstance(cols, list) and cols:
        return [c for c in cols if isinstance(c, str)]

    exp = rule.get("expectation") or {}
    if not isinstance(exp, dict):
        exp = {}

    if rule_type == "schema":
        rc = exp.get("required_columns")
        if isinstance(rc, list):
            return [c for c in rc if isinstance(c, str)]
        return []

    if rule_type == "freshness":
        ts = exp.get("ts_column")
        return [ts] if isinstance(ts, str) else []

    # most column-based rules
    col = exp.get("column")
    return [col] if isinstance(col, str) else []


def _normalize_payload(rule_type: str | None, rule: dict[str, Any]) -> dict[str, Any]:
    """
    Payload should NOT include column keys (we already normalize them separately).
    Use:
      - patch: params
      - canonical: expectation (minus column-ish keys)
    """
    payload = rule.get("params")
    if payload is None:
        payload = rule.get("expectation") or {}
    if not isinstance(payload, dict):
        return {}

    # remove column-ish keys so patch/canonical dedup works
    drop_keys = {"column", "columns", "required_columns", "ts_column"}
    cleaned = {k: v for k, v in payload.items() if k not in drop_keys}

    # Optional: for schema, required_columns are handled via columns, so payload usually empty
    return cleaned


def _signature(rule: dict[str, Any]) -> tuple:
    rule_type = rule.get("type") or rule.get("rule_type")
    if rule_type == "etl_validation":
        return (rule_type, _normalize_sql_refs(rule))
    cols = _normalize_cols(rule_type, rule)
    # Stable ordering
    cols_t = tuple(sorted(cols))
    payload = _normalize_payload(rule_type, rule)
    payload_t = _freeze(payload)
    return (rule_type, cols_t, payload_t)


def _reject(rule: dict[str, Any], reason: str) -> dict[str, Any]:
    out = dict(rule) if isinstance(rule, dict) else {"raw": rule}
    out["reject_reason"] = reason
    return out


def _validate_anomaly_params(params: dict[str, Any]) -> str | None:
    method = params.get("method")
    if method not in {"non_negative", "zscore", "iqr"}:
        return "anomaly_detection.params.method must be one of: non_negative, zscore, iqr"

    if method in {"zscore", "iqr"}:
        thr = params.get("threshold")
        try:
            thr_f = float(thr)
        except Exception:
            return f"anomaly_detection.params.threshold must be numeric for method={method}"
        if thr_f <= 0:
            return "anomaly_detection.params.threshold must be > 0"

    direction = params.get("direction")
    if direction is not None and direction not in {"low", "high", "both"}:
        return "anomaly_detection.params.direction must be one of: low, high, both"

    min_hard = params.get("min_hard")
    max_hard = params.get("max_hard")
    if min_hard is not None and max_hard is not None:
        try:
            if float(min_hard) > float(max_hard):
                return "anomaly_detection.params min_hard cannot be greater than max_hard"
        except Exception:
            return "anomaly_detection.params min_hard/max_hard must be numeric if provided"

    return None


def _is_business_non_negative_column(column: str, business_context: dict[str, Any] | None) -> bool:
    if not business_context:
        return False
    cols_desc = business_context.get("columns_description")
    if not isinstance(cols_desc, dict):
        return False

    raw_desc = cols_desc.get(column)
    if not isinstance(raw_desc, str):
        return False

    desc = raw_desc.lower()
    markers = (
        "non-negative",
        "non negative",
        "must be non-negative",
        "must be non negative",
        "cannot be negative",
        "can't be negative",
        ">= 0",
    )
    return any(m in desc for m in markers)


def validate_and_filter_ai_rules(
    *,
    ai_rules: list[dict[str, Any]],
    allowed_rule_types: list[str],
    dataset_columns: set[str],
    existing_rules: list[dict[str, Any]],
    max_rules_to_add: int = 15,
    min_ai_confidence: float | None = None,
    business_context: dict[str, Any] | None = None,
) -> PatchDecision:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    existing_sigs = {_signature(r) for r in existing_rules if isinstance(r, dict)}

    for r in ai_rules:
        if not isinstance(r, dict):
            rejected.append(_reject(r, "rule is not a dict"))
            continue

        rtype = r.get("rule_type") or r.get("type")
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
        col_rules = {
            "completeness",
            "uniqueness",
            "domain",
            "range",
            "date_not_in_future",
            "anomaly_detection",
        }
        if rtype in col_rules:
            col = r.get("column")
            if col is None and isinstance(r.get("columns"), list):
                legacy_cols = [c for c in r.get("columns", []) if isinstance(c, str) and c.strip()]
                if len(legacy_cols) == 1:
                    col = legacy_cols[0]
                    r["column"] = col
                elif len(legacy_cols) > 1:
                    rejected.append(_reject(r, "only single column supported in patch format"))
                    continue
            if not isinstance(col, str) or col.strip() == "":
                rejected.append(_reject(r, "missing/invalid column"))
                continue
            if col not in dataset_columns:
                rejected.append(_reject(r, f"column not in dataset: {col}"))
                continue

        params = r.get("params") or {}
        if params is not None and not isinstance(params, dict):
            rejected.append(_reject(r, "params must be dict"))
            continue

        if rtype == "domain":
            if "allowed_values" not in params and "max_distinct" not in params:
                rejected.append(_reject(r, "domain.params must include allowed_values (preferred)"))
                continue

        if rtype == "range":
            if "min" not in params and "max" not in params:
                rejected.append(_reject(r, "range.params must include min and/or max"))
                continue

        if rtype == "anomaly_detection":
            reason = _validate_anomaly_params(params)
            if reason is not None:
                rejected.append(_reject(r, reason))
                continue

            method = params.get("method")
            col = r.get("column")
            if (
                isinstance(col, str)
                and method in {"zscore", "iqr"}
                and _is_business_non_negative_column(col, business_context)
            ):
                # Business semantics override statistical method for hard non-negative columns.
                r = dict(r)
                r["params"] = {"method": "non_negative"}
                prior = str(r.get("rationale", "")).strip()
                r["rationale"] = (
                    (prior + " ") if prior else ""
                ) + f"Method normalized to non_negative from {method} by business context."

        # etl_validation: validate sql_ref shape
        if rtype == "etl_validation":
            exp = r.get("expectation") or {}
            sql_refs = exp.get("sql_ref")
            if not isinstance(sql_refs, list) or len(sql_refs) == 0:
                rejected.append(_reject(r, "etl_validation.expectation.sql_ref must be a non-empty list"))
                continue
            inline_count = 0
            valid_refs = True
            for ref in sql_refs:
                if not isinstance(ref, dict):
                    rejected.append(_reject(r, "each sql_ref item must be a dict"))
                    valid_refs = False
                    break
                if "file" not in ref and "inline_sql" not in ref:
                    rejected.append(_reject(r, "each sql_ref item must have 'file' or 'inline_sql'"))
                    valid_refs = False
                    break
                if "inline_sql" in ref:
                    inline_count += 1
            if not valid_refs:
                continue
            if inline_count > 1:
                rejected.append(_reject(r, "at most one inline_sql item is allowed per etl_validation rule"))
                continue

        sig = _signature(r)
        # <-- this was failing because sig contained unhashables
        if sig in existing_sigs:
            rejected.append(_reject(r, "duplicate rule (signature already exists)"))
            continue

        accepted.append(r)
        existing_sigs.add(sig)

        if len(accepted) >= max_rules_to_add:
            break

    return PatchDecision(accepted=accepted, rejected=rejected)
