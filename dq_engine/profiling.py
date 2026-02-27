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

        distinct = int(s.nunique(dropna=True))
        top = s.value_counts(dropna=True).head(max_top).to_dict()

        col_prof: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": nulls,
            "null_pct": (nulls / n * 100.0) if n else 0.0,
            "distinct": distinct,
            "top_values": top,
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
