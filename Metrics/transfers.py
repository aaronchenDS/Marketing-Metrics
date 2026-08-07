"""Transfer metrics (spec: Unanswered transfer rate -- notebook 11.7).

A transfer "attempt" is any of these events on a conversation: TransferStart,
TransferAttempt, CallTransfer, or a CallEnd event with reason == "TRANSFER".
A transfer is "answered" if that same conversationId also has a
TransferComplete event somewhere. Unanswered = attempted but never
completed. This is the notebook's exact definition (11.7), just re-weighted
here per restaurant/month instead of a single platform-wide number.

These events carry a conversationId but not a restaurantId or createdAt
directly, so data_prep.py joins them to a small conversationId ->
(restaurantId, dateCreated) lookup pulled from the conversations collection.
"""
import pandas as pd

from Metrics.common import percentile_groups

ATTEMPT_TYPES = {"TransferStart", "TransferAttempt", "CallTransfer"}


def build_monthly(transfer_events: pd.DataFrame, conv_lookup: pd.DataFrame,
                   restaurants: pd.DataFrame) -> pd.DataFrame:
    """One row per (restaurant, month): transfer attempts + unanswered count."""
    is_attempt = transfer_events["type"].isin(ATTEMPT_TYPES) | (
        (transfer_events["type"] == "CallEnd") & (transfer_events.get("reason") == "TRANSFER")
    )
    is_complete = transfer_events["type"] == "TransferComplete"

    attempt_ids = transfer_events.loc[is_attempt, "conversationId"].dropna().unique()
    complete_ids = set(transfer_events.loc[is_complete, "conversationId"].dropna().unique())

    attempts = pd.DataFrame({"conversationId": attempt_ids})
    attempts["is_unanswered"] = ~attempts["conversationId"].isin(complete_ids)

    merged = attempts.merge(conv_lookup, on="conversationId", how="left")
    merged = merged.dropna(subset=["restaurantId"])
    merged["restaurantId"] = merged["restaurantId"].astype(str)
    merged["dateCreated"] = pd.to_datetime(merged["dateCreated"], errors="coerce", utc=True)
    merged["year_month"] = merged["dateCreated"].dt.tz_convert(None).dt.to_period("M").astype(str)

    monthly = merged.groupby(["restaurantId", "year_month"]).agg(
        transfer_attempts=("conversationId", "size"),
        unanswered_n=("is_unanswered", "sum"),
    ).reset_index()

    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")
    return monthly


# --- App-side slicing helpers ----------------------------------------------

def platform_kpis(view: pd.DataFrame) -> dict:
    attempts = view["transfer_attempts"].sum()
    unanswered = view["unanswered_n"].sum()
    return {
        "transfer_attempts": int(attempts),
        "unanswered_transfer_rate": unanswered / attempts if attempts else float("nan"),
    }


def _restaurant_rollup(view: pd.DataFrame) -> pd.DataFrame:
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        transfer_attempts=("transfer_attempts", "sum"),
        unanswered_n=("unanswered_n", "sum"),
    )
    g["unanswered_transfer_rate"] = g["unanswered_n"] / g["transfer_attempts"].replace(0, float("nan"))
    return g


def flagged_restaurants(view: pd.DataFrame, threshold: float = 0.25, min_attempts: int = 5) -> pd.DataFrame:
    """Restaurants whose unanswered-transfer rate is above `threshold`
    (spec: "flag customers above 25%")."""
    g = _restaurant_rollup(view)
    eligible = g[g["transfer_attempts"] >= min_attempts]
    empty_cols = ["name", "transfer_attempts", "unanswered_transfer_rate"]
    if eligible.empty:
        return eligible.reindex(columns=empty_cols)
    flagged = eligible[eligible["unanswered_transfer_rate"] > threshold]
    return (
        flagged.sort_values("unanswered_transfer_rate", ascending=False)[empty_cols]
        .reset_index(drop=True)
    )


def restaurant_percentiles(view: pd.DataFrame):
    """Bottom-25% / top-90% restaurant lists by unanswered-transfer rate."""
    g = _restaurant_rollup(view)
    return percentile_groups(g, "unanswered_transfer_rate")
