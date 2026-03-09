from __future__ import annotations

import pandas as pd


def summarize_run(run_id: str, all_results: pd.DataFrame) -> dict[str, object]:
    total = len(all_results)
    failed = int((all_results["status"] == "fail").sum())
    failed_critical = int(
        ((all_results["status"] == "fail") & (all_results["severity"] == "critical")).sum()
    )
    score = 100.0 if total == 0 else max(0.0, 100.0 - 100.0 * (failed / total))
    status = "pass"
    if failed_critical > 0:
        status = "fail"
    elif failed > 0:
        status = "warn"
    return {
        "run_id": run_id,
        "status": status,
        "score": round(float(score), 2),
        "total_rules": int(total),
        "failed": failed,
        "failed_critical": failed_critical,
    }
