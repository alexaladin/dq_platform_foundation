# DQ Platform — Local Foundation (Portable to Databricks)

This repository is a **local-first** Data Quality prototype structured as a product foundation.
It is designed to be migrated to Azure/Databricks later with minimal rewrite:
- deterministic rule execution engine (portable logic)
- rules-as-code (YAML) + JSON schema validation
- run outputs (append-only artifacts) with bad-record samples
- CLI runner and unit tests

## Folder structure
- `data/raw/` — local input data (CSV)
- `dq_engine/` — core deterministic DQ engine (Python package)
- `dq_registry/` — rules-as-code YAML + schema
- `dq_results/` — outputs (runs, rule_results, bad_samples, summaries)
- `notebooks/` — Jupyter entry points (thin UI layer)
- `config/` — datasets and policy configs
- `run_local.py` — CLI runner

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: run tests
pytest -q

# Run DQ
python run_local.py --project_root .
```
Outputs will be written under `dq_results/`.

## Notebooks (optional)
Open in order:
- `notebooks/00_setup.ipynb`
- `notebooks/01_profiling_and_suggest_rules.ipynb`
- `notebooks/02_execute_checks.ipynb`
- `notebooks/03_reporting.ipynb`

## Migration to Databricks (later)
- Replace pandas dataset loaders with Spark/Delta readers.
- Persist registry/results in Delta tables (instead of local CSV artifacts).
- Keep rule definitions (YAML) and the check semantics unchanged.