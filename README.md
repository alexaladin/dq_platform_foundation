# DQ Platform Foundation

Local-first Data Quality platform for profiling datasets, managing rules as code, executing deterministic checks, and optionally using AI to suggest additional rules.
The project is designed to stay portable to Databricks/Spark with minimal rewrite of business logic.

## What This Project Does

- Loads datasets from `config/datasets.yaml`
- Loads rulesets from `dq_registry/rulesets/*.yaml`
- Executes deterministic DQ checks (`schema`, `completeness`, `uniqueness`, `range`, `domain`, `date_not_in_future`, `freshness`, `referential_integrity`)
- Writes auditable run artifacts to `dq_results/`
- Supports AI-assisted rule suggestion (Azure OpenAI / CodeMie / Mock) in patch mode, including advisory `anomaly_detection` rules
- Merges accepted AI rules safely through guardrails

## Repository Structure

- `run_local.py`: main local execution entrypoint for deterministic checks
- `scripts/ai_suggested_rules.py`: rule suggestion pipeline (deterministic + AI patch)
- `config/`
  - `datasets.yaml`: dataset sources + deterministic schema/freshness config
  - `dq_standards.yaml`: standards and AI/key-detection settings
- `dq_engine/`
  - `checks.py`: check implementations
  - `execution.py`: rule orchestration and results writing
  - `registry.py`: ruleset loading
  - `ai_patch_guardrails.py`: AI rule validation/filtering
  - `rules_merge.py`: patch merge into canonical ruleset shape
  - `profiling.py`: column profiling
  - `scoring.py`: run-level scoring
- `dq_ai/`
  - `provider_azure_openai.py`: Azure provider
  - `provider_codemie_assistant.py`: CodeMie provider
  - `provider_mock.py`: mock provider for local testing
  - `payload_builder.py`: curated per-rule-type column candidates for AI
- `dq_registry/`
  - `rulesets/`: rules-as-code YAML
  - `schemas/`: schema definitions
- `dq_results/`: run outputs, AI suggestion artifacts, summaries
- `tests/`: unit tests
- `notebooks/`: optional interactive workflow notebooks

## High-Level Architecture

1. **Config Layer**
   - `config/datasets.yaml` defines each dataset and deterministic controls:
     - `source_location`
     - `required_columns` (for deterministic schema rule)
     - `ts_load_column` + `freshness_max_age_days` (for deterministic freshness rule)

2. **Deterministic Engine**
   - Rules are loaded from YAML and executed in `dq_engine/execution.py`.
   - Each rule maps to a check in `dq_engine/checks.py`.
   - Failures can produce bad-record samples.

3. **AI Suggestion Pipeline (Optional)**
   - `scripts/ai_suggested_rules.py` profiles data, injects deterministic baseline rules, then asks an AI provider for `rules_to_add`.
   - `dq_ai/payload_builder.py` sends curated `column_candidates` instead of full profiling.
   - Guardrails validate and filter AI output before merge.

4. **Artifacts & Auditability**
   - Rule results: `dq_results/rule_results/`
   - Bad samples: `dq_results/bad_samples/`
   - Run summaries: `dq_results/run_summaries/`
  - AI inputs/raw/accepted/rejected patches: `dq_results/ai_suggestions/`
  - Advisory anomaly artifacts: `...__suggest_rules_rationale.txt`, `...__anomaly_samples.csv`, `...__anomaly_summary.json`

## Onboarding (New Team Members)

### 1) Prerequisites

- Python 3.10+ (project targets 3.10/3.11)
- Git
- Windows PowerShell or Bash

### 2) Environment Setup

```bash

Windows PowerShell:

Bash:

Install dependencies:

3) Validate Setup
Optional formatting/linting:

Core Workflows
A) Run Deterministic DQ for All Configured Datasets
Useful flags:

--run_id <id>
--datasets_config config/datasets.yaml
--rulesets_dir dq_registry/rulesets
--results_dir dq_results
--max_samples 100
B) Generate Suggested Rules for One Dataset
Deterministic only:

Mock AI:

Azure AI:

CodeMie AI:

C) Review Outputs
Suggested ruleset: dq_registry/rulesets/<dataset>.suggested.yaml
AI audit trail: dq_results/ai_suggestions/*
Deterministic run outputs: dq_results/rule_results/, dq_results/run_summaries/, dq_results/bad_samples/
Config Reference
config/datasets.yaml
Per dataset supported keys:

dataset_id
source_type (currently csv)
source_location
owner_team
data_owner
sla_tier
required_columns (optional but recommended)
ts_load_column (required for deterministic freshness)
freshness_max_age_days (required for deterministic freshness)
freshness_severity (optional)
AI Provider Environment Variables
Azure
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION
CodeMie
CODEMIE_API_DOMAIN
CODEMIE_ASSISTANT_ID
CODEMIE_USERNAME
CODEMIE_PASSWORD
CODEMIE_AUTH_REALM
CODEMIE_AUTH_SERVER_URL
Optional:
CODEMIE_AUTH_CLIENT_ID (default: codemie-sdk)
CODEMIE_VERIFY_SSL (default: true)
CODEMIE_TIMEOUT_S (default: 90)
Common Troubleshooting
TypeError: Rule.__init__() got an unexpected keyword argument ...

Ensure loader uses filtered known fields in dq_engine/registry.py.
Missing deterministic freshness rule

Confirm dataset has both ts_load_column and freshness_max_age_days in config/datasets.yaml.
AI not suggesting date rules

Confirm date_not_in_future is in allowed rule types and date-like columns are present in column_candidates.date_not_in_future.
Roadmap / Portability Notes
To migrate to Databricks:

Swap pandas CSV readers for Spark/Delta readers
Persist registry/results in tables
Keep rule semantics and guardrails unchanged
Reuse most of dq_engine and dq_ai logic with minimal interface changes
python -m venv .venv
