"""Known-bad reference: returns the input unchanged (passes leak_free, fails normalises_nontrivially)."""

import pandas as pd


def normalise(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.copy()
