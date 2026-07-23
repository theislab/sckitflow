from collections.abc import Collection

import pandas as pd

__all__ = ["convert_to_categorical_in_place"]


def convert_to_categorical_in_place(df: pd.DataFrame, cols: Collection[str] | None) -> None:
    if cols is None:
        return
    for c in cols:
        if c in df.columns and not hasattr(df[c], "cat"):
            df[c] = df[c].astype("category")
