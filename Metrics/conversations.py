"""Conversation metrics.

Three kinds of function live here:

1. build_monthly / build_callers_monthly / build_simultaneous_monthly ->
   called by data_prep.py to turn raw conversation rows into small
   per-restaurant-per-month tables (the snapshots).
2. the platform_kpis / monthly_totals / top_restaurants / *_percentiles
   helpers -> called by app.py to slice those snapshots for whatever the
   user selected.

Key idea you'll reuse everywhere: the snapshot stores ADDITIVE quantities
(counts and sums), never pre-divided averages. Rates and averages are derived
in the app by summing the pieces and dividing. That's what lets the app
re-aggregate correctly for any month range or restaurant subset.
"""
import numpy as np
import pandas as pd

from Metrics.common import percentile_groups


def build_monthly(conv: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw conversations into one row per (restaurant, month)."""
    conv = conv.copy()
    conv["restaurantId"] = conv["restaurantId"].astype(str)

    conv["dateCreated"] = pd.to_datetime(conv["dateCreated"], errors="coerce", utc=True)
    conv["dateEnded"] = pd.to_datetime(conv["dateEnded"], errors="coerce", utc=True)
    conv["year_month"] = conv["dateCreated"].dt.tz_convert(None).dt.to_period("M").astype(str)

    # Call length in seconds, but only trust values between 0 and 4 hours.
    dur = (conv["dateEnded"] - conv["dateCreated"]).dt.total_seconds()
    conv["duration_s"] = dur.where(dur.between(0, 4 * 3600))

    conv["sent_to_host"] = conv["convoSentToHost"].astype(str).str.lower().eq("true")
    conv["score"] = pd.to_numeric(conv.get("conversationScore"), errors="coerce")

    grouped = conv.groupby(["restaurantId", "year_month"])
    monthly = grouped.agg(
        conversations=("restaurantId", "size"),
        duration_sum=("duration_s", "sum"),
        duration_n=("duration_s", "count"),      # count ignores NaN -> valid durations only
        sent_to_host_n=("sent_to_host", "sum"),
        score_sum=("score", "sum"),
        score_n=("score", "count"),
    ).reset_index()

    # Attach restaurant names (restaurants._id -> name).
    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")

    return monthly


def build_callers_monthly(conv: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """New vs. repeat callers (notebook 11.4). Requires a `phoneFrom` column.

    "New" = the first call ever seen from that phone number, at that
    restaurant. Everything after is "Repeat". Also tracks the pieces needed
    for "average calls per repeat caller per month" as a sum/count pair
    (restaurant-month groups with >1 call from the same phone, and their
    total call count), so it re-aggregates correctly for any selection.
    """
    conv = conv.dropna(subset=["restaurantId", "phoneFrom", "dateCreated"]).copy()
    conv["restaurantId"] = conv["restaurantId"].astype(str)
    conv["dateCreated"] = pd.to_datetime(conv["dateCreated"], errors="coerce", utc=True)
    conv["year_month"] = conv["dateCreated"].dt.tz_convert(None).dt.to_period("M").astype(str)
    conv = conv.sort_values(["restaurantId", "phoneFrom", "dateCreated"])

    # New = first-ever call from that phone number to that restaurant
    # (across all time, not reset each month).
    conv["caller_seq"] = conv.groupby(["restaurantId", "phoneFrom"]).cumcount() + 1
    conv["is_new"] = conv["caller_seq"].eq(1)

    monthly = conv.groupby(["restaurantId", "year_month"]).agg(
        new_callers=("is_new", "sum"),
        conversations=("restaurantId", "size"),
    ).reset_index()
    monthly["repeat_callers"] = monthly["conversations"] - monthly["new_callers"]

    # Repeat-frequency pieces: per (restaurant, phone, month) groups with
    # more than one call, track how many such groups and their total calls.
    group_calls = (
        conv.groupby(["restaurantId", "phoneFrom", "year_month"])
        .size()
        .reset_index(name="calls")
    )
    repeat_groups = group_calls[group_calls["calls"] > 1]
    repeat_freq = repeat_groups.groupby(["restaurantId", "year_month"]).agg(
        repeat_group_call_sum=("calls", "sum"),
        repeat_group_n=("calls", "size"),
    ).reset_index()

    monthly = monthly.merge(repeat_freq, on=["restaurantId", "year_month"], how="left")
    monthly[["repeat_group_call_sum", "repeat_group_n"]] = (
        monthly[["repeat_group_call_sum", "repeat_group_n"]].fillna(0)
    )

    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")
    return monthly


def build_simultaneous_monthly(conv: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """Simultaneous-call rate (notebook 11.6): a call counts as simultaneous
    if it starts before the latest-ending prior call (at that restaurant)
    has finished. Overlap is detected across each restaurant's FULL
    timeline (not chunked by month) so it isn't blind to calls that
    straddle a month boundary, then the result is bucketed by month.
    """
    conv = conv.dropna(subset=["restaurantId", "dateCreated", "dateEnded"]).copy()
    conv["restaurantId"] = conv["restaurantId"].astype(str)
    conv["dateCreated"] = pd.to_datetime(conv["dateCreated"], errors="coerce", utc=True)
    conv["dateEnded"] = pd.to_datetime(conv["dateEnded"], errors="coerce", utc=True)
    conv["year_month"] = conv["dateCreated"].dt.tz_convert(None).dt.to_period("M").astype(str)
    conv = conv.sort_values(["restaurantId", "dateCreated"])

    conv["latest_prior_end"] = conv.groupby("restaurantId")["dateEnded"].cummax().shift()
    boundary = conv["restaurantId"].ne(conv["restaurantId"].shift())
    conv.loc[boundary, "latest_prior_end"] = pd.NaT
    conv["is_simultaneous"] = conv["dateCreated"] < conv["latest_prior_end"]

    monthly = conv.groupby(["restaurantId", "year_month"]).agg(
        total_calls=("restaurantId", "size"),
        simultaneous_calls=("is_simultaneous", "sum"),
    ).reset_index()

    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")
    return monthly


# --- App-side slicing helpers ---------------------------------------------

def platform_kpis(view: pd.DataFrame) -> dict:
    """Headline numbers for the selected slice, correctly weighted.

    Mirrors the notebook's 11.1 conversation_summary + 11.3 duration + 11.5
    sent-to-host, but re-weighted from the additive columns for any subset.
    """
    conversations = int(view["conversations"].sum())
    duration_n = view["duration_n"].sum()
    score_n = view["score_n"].sum()
    return {
        "conversations": conversations,
        "active_restaurants": int(view["restaurantId"].nunique()),
        "total_time_hours": view["duration_sum"].sum() / 3600,
        "avg_duration_s": view["duration_sum"].sum() / duration_n if duration_n else float("nan"),
        "sent_to_host_rate": view["sent_to_host_n"].sum() / conversations if conversations else float("nan"),
        "avg_score": view["score_sum"].sum() / score_n if score_n else float("nan"),
    }


def monthly_totals(view: pd.DataFrame) -> pd.DataFrame:
    """Total conversations per month, for the trend line."""
    return (
        view.groupby("year_month", as_index=False)["conversations"]
        .sum()
        .sort_values("year_month")
    )


def _restaurant_rollup(view: pd.DataFrame) -> pd.DataFrame:
    """Collapse the month dimension: one row per restaurant, with derived rates."""
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        conversations=("conversations", "sum"),
        duration_sum=("duration_sum", "sum"),
        duration_n=("duration_n", "sum"),
        sent_to_host_n=("sent_to_host_n", "sum"),
        score_sum=("score_sum", "sum"),
        score_n=("score_n", "sum"),
    )
    g["avg_duration_s"] = g["duration_sum"] / g["duration_n"].replace(0, np.nan)
    g["sent_to_host_rate"] = g["sent_to_host_n"] / g["conversations"].replace(0, np.nan)
    g["avg_score"] = g["score_sum"] / g["score_n"].replace(0, np.nan)
    return g


def top_restaurants(view: pd.DataFrame, metric: str = "conversations",
                    n: int = 10, min_conversations: int = 30) -> pd.DataFrame:
    """Top N restaurants by a chosen metric. Rate metrics require a min volume."""
    g = _restaurant_rollup(view)
    if metric in ("sent_to_host_rate", "avg_score", "avg_duration_s"):
        g = g[g["conversations"] >= min_conversations]
    display_cols = ["name", "conversations", "avg_duration_s", "sent_to_host_rate", "avg_score"]
    return (
        g.sort_values(metric, ascending=False)[display_cols]
        .head(n)
        .reset_index(drop=True)
    )


def restaurant_percentiles(view: pd.DataFrame, metric: str = "conversations"):
    """Bottom-25% / top-90% restaurant lists for `metric` (spec's percentile
    spread) -- metric can be "conversations" or "sent_to_host_rate"
    (resolved-by-Hostie rate)."""
    g = _restaurant_rollup(view)
    return percentile_groups(g, metric)


# --- New vs. repeat callers -------------------------------------------------

def caller_kpis(view: pd.DataFrame) -> dict:
    new_callers = view["new_callers"].sum()
    repeat_callers = view["repeat_callers"].sum()
    total = new_callers + repeat_callers
    repeat_group_call_sum = view["repeat_group_call_sum"].sum()
    repeat_group_n = view["repeat_group_n"].sum()
    return {
        "new_callers": int(new_callers),
        "repeat_callers": int(repeat_callers),
        "new_caller_share": new_callers / total if total else float("nan"),
        "repeat_caller_share": repeat_callers / total if total else float("nan"),
        "avg_calls_per_repeat_caller_month": (
            repeat_group_call_sum / repeat_group_n if repeat_group_n else float("nan")
        ),
    }


def new_caller_trend(view: pd.DataFrame) -> pd.DataFrame:
    """New callers per month + month-over-month growth (notebook 11.4)."""
    g = (
        view.groupby("year_month", as_index=False)["new_callers"]
        .sum()
        .sort_values("year_month")
    )
    g["mom_growth"] = g["new_callers"].pct_change()
    return g


# --- Simultaneous calls ------------------------------------------------------

def simultaneous_kpis(view: pd.DataFrame) -> dict:
    total = view["total_calls"].sum()
    simultaneous = view["simultaneous_calls"].sum()
    return {
        "total_calls": int(total),
        "simultaneous_calls": int(simultaneous),
        "simultaneous_call_rate": simultaneous / total if total else float("nan"),
    }


def _simultaneous_rollup(view: pd.DataFrame) -> pd.DataFrame:
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        total_calls=("total_calls", "sum"),
        simultaneous_calls=("simultaneous_calls", "sum"),
    )
    g["simultaneous_call_rate"] = g["simultaneous_calls"] / g["total_calls"].replace(0, np.nan)
    return g


def simultaneous_percentiles(view: pd.DataFrame):
    """Bottom-25% / top-90% restaurant lists by simultaneous-call rate."""
    g = _simultaneous_rollup(view)
    return percentile_groups(g, "simultaneous_call_rate")
