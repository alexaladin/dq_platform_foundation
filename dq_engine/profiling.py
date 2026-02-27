from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_object_dtype

def profile_df(df: pd.DataFrame, max_top: int = 10) -> Dict[str, Any]:
    prof: Dict[str, Any] = {"row_count": int(len(df)), "columns": {}}
    n = len(df)

    for c in df.columns:
        s = df[c]

        # count nulls + blanks for string/object
        if is_string_dtype(s) or is_object_dtype(s):
            blanks = s.astype("string").str.strip().eq("").sum()
            nulls = int(s.isna().sum() + blanks)
        else:
            nulls = int(s.isna().sum())

        top = s.value_counts(dropna=True).head(max_top).to_dict()

        non_null_count = n - nulls
        distinct_count = int(s.nunique(dropna=True))
        distinct_ratio_non_null = distinct_count / non_null_count
        max_dup_count = int(s.value_counts().max())
        duplicate_count = non_null_count - distinct_count

        col_prof: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": nulls,
            "null_pct": float(nulls / n) if n else 0.0,
            "distinct_count": distinct_count,
            "top_values": top,
            "non_null_count": int(non_null_count),
            "distinct_ratio_non_null": float(distinct_ratio_non_null),
            "max_dup_count": max_dup_count,
            "duplicate_count": duplicate_count
        }

        # numeric stats (safe for extension dtypes)
        if is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().any():
                col_prof.update({
                    "min": float(np.nanmin(sn.values)),
                    "max": float(np.nanmax(sn.values)),
                    "mean": float(np.nanmean(sn.values)),
                })
            else:
                col_prof.update({"min": None, "max": None, "mean": None})

        prof["columns"][c] = col_prof

    return prof
