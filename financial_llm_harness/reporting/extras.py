"""Optional results files: ``consistency.csv`` (§T2.2) and ``detection.csv`` (§T2.1).

They exist only for pairwise or ranking experiments and for type D items. No
V1 slice has either; the families that do arrive at M9a, which fills this in.
"""

from __future__ import annotations

from financial_llm_harness.reporting.tables import RunData


def optional_tables(rd: RunData) -> dict[str, str]:
    """File name → CSV text for the optional tables this run has (none before M9a)."""
    del rd
    return {}
