import json
from pathlib import Path
import yaml
from jsonschema import validate

def test_rulesets_against_schema():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root/"dq_registry/schemas/ruleset.schema.json").read_text(encoding="utf-8"))
    rules_dir = root/"dq_registry/rulesets"
    for p in rules_dir.glob("*.yaml"):
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        validate(instance=doc, schema=schema)
