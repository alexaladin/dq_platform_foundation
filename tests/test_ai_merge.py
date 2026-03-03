from dq_engine.rules_merge import merge_rules_to_add


def test_merge_appends_rules_with_ids_and_audit_fields():
    doc = {"dataset_id": "x", "ruleset_version": 1, "rules": []}
    rules = [
        {
            "type": "domain",
            "columns": ["status"],
            "severity": "medium",
            "params": {"allowed_values": ["A", "B"]},
            "confidence": 0.9,
            "rationale": "enum-like",
            "evidence_used": ["distinct_count=2"],
        }
    ]
    merge_rules_to_add(ruleset_doc=doc, rules_to_add=rules, suggested_by="ai_patcher")

    assert len(doc["rules"]) == 1
    r0 = doc["rules"][0]
    assert r0["rule_id"].startswith("domain_") or r0["rule_id"].startswith("D")
    assert r0["suggested_by"] == "ai_patcher"
    assert r0["ai_confidence"] == 0.9
    assert r0["ai_rationale"] == "enum-like"
