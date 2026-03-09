# AI Patcher (Patch-Mode)

## Goal
Use AI to propose DQ checks as *additions only* (rules_to_add), never a full ruleset.

## Contract
AI returns JSON:
- rules_to_add: list of rule objects
- rationale: overall summary

Rule object:
- type: must be in allowed_rule_types
- columns: must exist in dataset (for column-based rules)
- severity: high|medium|low
- params: dict (rule-specific)
- confidence: float 0..1
- rationale: short explanation
- evidence_used: list of strings referencing profiling metrics

## Guardrails
Before merge we:
- validate type allowlist
- validate columns existence
- validate params by type
- dedupe by (type + columns + params)
- cap max_rules_to_add
- optional min_confidence threshold
- generate rule_id ourselves
- append-only merge

## How to run
Deterministic only:
python scripts/ai_suggest_rules.py --dataset invent_location --ai off

Mock AI patch:
python scripts/ai_suggest_rules.py --dataset invent_location --ai mock

Azure AI patch:
set AZURE_OPENAI_ENDPOINT=...
set AZURE_OPENAI_API_KEY=...
set AZURE_OPENAI_DEPLOYMENT=...
python scripts/ai_suggest_rules.py --dataset invent_location --ai azure --min_ai_confidence 0.6

Artifacts saved to:
dq_results/ai_suggestions/
