"""Shared helpers used across metrics modules.

percentile_groups() is the notebook's own pattern (used for total
conversations, covers, simultaneous-call rate, speak-to-human rate, etc.):
split a restaurant-level table into the bottom-25% and top-90% (>=90th
percentile) groups by some metric, each sorted so the most extreme
restaurant is first.
"""
import pandas as pd


def percentile_groups(table: pd.DataFrame, metric: str):
    """Return (bottom_25pct, top_90pct) restaurant tables for `metric`.

    bottom_25pct: rows at or below the 25th percentile, ascending (worst/lowest first).
    top_90pct: rows at or above the 90th percentile, descending (best/highest first).
    """
    if table.empty:
        return table, table
    bottom_cutoff = table[metric].quantile(0.25)
    top_cutoff = table[metric].quantile(0.90)

    bottom_table = table[table[metric] <= bottom_cutoff].sort_values(metric, ascending=True)
    top_table = table[table[metric] >= top_cutoff].sort_values(metric, ascending=False)

    return bottom_table, top_table
