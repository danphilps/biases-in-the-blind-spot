"""Known-good reference: expanding-window z-score per asset and feature (passes all four checks)."""

import pandas as pd


def normalise(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = panel.groupby(level=1, sort=False, group_keys=False)
    mean = grouped.transform(lambda df: df.expanding().mean())
    std = grouped.transform(lambda df: df.expanding().std())
    return (panel - mean) / std
