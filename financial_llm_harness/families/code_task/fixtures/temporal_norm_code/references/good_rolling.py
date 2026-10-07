"""Known-good reference: 252-date rolling z-score per asset and feature (passes all four checks)."""

import pandas as pd

WINDOW = 252


def normalise(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = panel.groupby(level=1, sort=False, group_keys=False)
    mean = grouped.transform(lambda df: df.rolling(WINDOW, min_periods=WINDOW).mean())
    std = grouped.transform(lambda df: df.rolling(WINDOW, min_periods=WINDOW).std())
    return (panel - mean) / std
