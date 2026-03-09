from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


def _match_any(patterns: list[str], s: str) -> list[str]:
    hits = []
    for p in patterns or []:
        if re.search(p, s, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class KeyEvidence:
    row_count: int
    non_null_count: int
    null_pct: float
    distinct_count: int
    distinct_ratio_non_null: float
    duplicate_count: int
    max_dup_count: int | None
    name_signals: dict[str, list[str]]


@dataclass
class KeyCandidate:
    candidate_type: str  # "single"
    columns: list[str]
    confidence: float
    evidence: KeyEvidence
    recommended_rules: list[dict[str, Any]]
    notes: list[str]


def _compute_confidence_single(
    col_name: str,
    col_prof: dict[str, Any],
    kd: dict[str, Any],
) -> tuple[float, dict[str, list[str]], list[str]]:
    weights = (kd.get("scoring", {}) or {}).get("weights", {}) or {}
    floors = (kd.get("scoring", {}) or {}).get("floors", {}) or {}

    null_pct = float(col_prof.get("null_pct", 1.0))  # ожидаем 0..1 # 0
    non_null = int(col_prof.get("non_null_count", 0))  # 11
    distinct = int(col_prof.get("distinct_count", col_prof.get("distinct", 0)))
    # 10
    dr = col_prof.get("distinct_ratio_non_null")
    if dr is None:
        dr = (distinct / non_null) if non_null > 0 else 0.0
    dr = float(dr)

    dup_count = max(0, non_null - distinct)  # 11 - 10 = 1
    max_dup_count = col_prof.get("max_dup_count")  # 2

    # Name hints
    name_hints = kd.get("name_hints", {}) or {}
    use_name_hints = bool(kd.get("use_name_hints", True))
    pos_hits = (
        _match_any(name_hints.get("positive_patterns", []), col_name) if use_name_hints else []
    )
    neg_hits = (
        _match_any(name_hints.get("negative_patterns", []), col_name) if use_name_hints else []
    )
    name_signals = {"positive": pos_hits, "negative": neg_hits}

    notes: list[str] = []

    exact_unique = dup_count == 0
    distinct_ratio_min = float(kd.get("distinct_ratio_min", 0.999))
    null_pct_max = float(kd.get("null_pct_max", 0.02))

    allow_near_unique = bool(kd.get("allow_near_unique", True))
    near_unique_cfg = kd.get("near_unique", {}) or {}
    max_duplicates = int(near_unique_cfg.get("max_duplicates", 0))
    max_duplicate_rate = float(near_unique_cfg.get("max_duplicate_rate", 0.0))

    dup_rate = (dup_count / non_null) if non_null > 0 else 1.0
    near_unique_ok = (
        allow_near_unique
        and (dup_count <= max_duplicates if max_duplicates > 0 else True)
        and (dup_rate <= max_duplicate_rate if max_duplicate_rate > 0 else True)
    )

    score = 0.0
    if exact_unique:
        score += float(weights.get("exact_unique", 0.6))
    elif near_unique_ok:
        score += float(weights.get("near_unique", 0.4))
        notes.append("near-unique (within near-unique thresholds)")
    else:
        notes.append("duplicates exceed near-unique thresholds")

    # distinct ratio scaled contribution
    dr_contrib = (dr - distinct_ratio_min) / max(1e-9, (1.0 - distinct_ratio_min))
    dr_contrib = _clamp(dr_contrib, 0.0, 1.0)
    score += float(weights.get("distinct_ratio", 0.2)) * dr_contrib

    # low nulls scaled contribution (0 nulls => 1.0, at null_pct_max => 0.0)
    null_contrib = 1.0 - (null_pct / max(1e-9, null_pct_max))
    null_contrib = _clamp(null_contrib, 0.0, 1.0)
    score += float(weights.get("low_nulls", 0.1)) * null_contrib

    if pos_hits:
        score += float(weights.get("positive_name_hint", 0.1))
    if neg_hits:
        score += float(weights.get("negative_name_hint", -0.2))

    score = _clamp(score, 0.0, 1.0)

    min_conf = float(floors.get("min_confidence_to_return", 0.0))
    if score < min_conf:
        notes.append(f"below min_confidence_to_return ({min_conf})")

    if max_dup_count is not None and int(max_dup_count) > 1:
        notes.append(f"max_dup_count={int(max_dup_count)}")
    return score, name_signals, notes


def suggest_key_candidates(
    profile: dict[str, Any],
    dq_standards: dict[str, Any],
    existing_rules: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    existing_rules = existing_rules or []

    kd = ((dq_standards.get("suggestions") or {}).get("key_detection")) or {}
    if not kd or not kd.get("enabled", True):
        return []

    row_count = int(profile.get("row_count", 0))
    cols = profile.get("columns") or {}

    max_candidates = int(kd.get("max_candidates", 3))
    # min_non_null = int(kd.get("min_non_null", 0))
    min_non_null = row_count * float(kd.get("distinct_ratio_min", 0.999))
    null_pct_max = float(kd.get("null_pct_max", 0.02))
    distinct_ratio_min = float(kd.get("distinct_ratio_min", 0.999))
    min_conf = float(
        ((kd.get("scoring", {}) or {}).get("floors", {}) or {}).get("min_confidence_to_return", 0.0)
    )

    # Candidate pool filter
    pool: list[tuple[str, dict[str, Any]]] = []
    for c, cp in cols.items():
        non_null = int(cp.get("non_null_count", 0))
        null_pct = float(cp.get("null_pct", 1.0))
        distinct = int(cp.get("distinct_count", cp.get("distinct", 0)))
        dr = cp.get("distinct_ratio_non_null")
        if dr is None:
            dr = (distinct / non_null) if non_null > 0 else 0.0
        dr = float(dr)

        if non_null < min_non_null:
            continue
        if null_pct > null_pct_max:
            continue

        near_cfg = kd.get("near_unique", {}) or {}
        near_dr_min = float(near_cfg.get("distinct_ratio_min", distinct_ratio_min))
        allow_near = bool(kd.get("allow_near_unique", True))
        if dr < distinct_ratio_min:
            #  если не проходит строгий порог, то может пройти как near-unique
            if not allow_near or dr < near_dr_min:
                continue

        pool.append((c, cp))
    candidates: list[KeyCandidate] = []
    for col_name, col_prof in pool:
        conf, name_signals, notes = _compute_confidence_single(col_name, col_prof, kd)
        if conf < min_conf:
            continue

        non_null = int(col_prof.get("non_null_count", 0))
        distinct = int(col_prof.get("distinct_count", col_prof.get("distinct", 0)))
        dr = col_prof.get("distinct_ratio_non_null")
        if dr is None:
            dr = (distinct / non_null) if non_null > 0 else 0.0

        dup_count = max(0, non_null - distinct)

        evidence = KeyEvidence(
            row_count=row_count,
            non_null_count=non_null,
            null_pct=float(col_prof.get("null_pct", 1.0)),
            distinct_count=distinct,
            distinct_ratio_non_null=float(dr),
            duplicate_count=dup_count,
            max_dup_count=col_prof.get("max_dup_count"),
            name_signals=name_signals,
        )

        recommended = [
            {"rule_type": "uniqueness", "column": col_name, "severity": "high"},
            {"rule_type": "completeness", "column": col_name, "severity": "high"},
        ]

        candidates.append(
            KeyCandidate(
                candidate_type="single",
                columns=[col_name],
                confidence=float(conf),
                evidence=evidence,
                recommended_rules=recommended,
                notes=notes,
            )
        )

    candidates_sorted = sorted(candidates, key=lambda x: x.confidence, reverse=True)[
        :max_candidates
    ]
    return [asdict(c) for c in candidates_sorted]
