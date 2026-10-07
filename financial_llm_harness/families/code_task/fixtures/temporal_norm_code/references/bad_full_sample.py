"""Known-bad reference: full-sample z-score per asset and feature (fails only leak_free)."""

import pandas as pd


def normalise(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = panel.groupby(level=1, sort=False)
    return (panel - grouped.transform("mean")) / grouped.transform("std")
