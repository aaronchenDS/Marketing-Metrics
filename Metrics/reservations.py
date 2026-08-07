"""Reservation metrics (spec sections 6-8: reservations, covers, cancellations,
edits -- notebook 11.16.1).

Reservations live in the events export as three event types:
- ReservationConfirm -> a completed reservation. Has businessId, createdAt,
  partySize (covers), and service (the booking platform, e.g. OpenTable).
- ReservationCancel -> a cancellation. Same businessId/createdAt, no
  partySize/service.
- ReservationEdit -> an edit to an existing reservation. Deliberately NOT
  ReservationEditFailed (a distinct event, excluded) -- confirmed exact
  event names via Compass.

Same additive-snapshot idea as conversations.py: build_monthly stores counts
and sums per (restaurant, month); rates/averages are derived in the app by
summing those pieces over whatever month range or restaurant subset the user
picks, so re-aggregating is always correct (never an "average of averages").
"""
import numpy as np
import pandas as pd

from Metrics.common import percentile_groups


def build_monthly(events: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """One row per (restaurant, month): reservations, covers, cancellations, edits."""
    events = events.copy()
    events["businessId"] = events["businessId"].astype(str)
    events["createdAt"] = pd.to_datetime(events["createdAt"], errors="coerce", utc=True)
    events["year_month"] = events["createdAt"].dt.tz_convert(None).dt.to_period("M").astype(str)
    events["partySize"] = pd.to_numeric(events.get("partySize"), errors="coerce")

    confirmed = events[events["type"] == "ReservationConfirm"]
    cancelled = events[events["type"] == "ReservationCancel"]
    edited = events[events["type"] == "ReservationEdit"]

    conf_g = confirmed.groupby(["businessId", "year_month"]).agg(
        reservations=("businessId", "size"),
        covers=("partySize", "sum"),
    ).reset_index()

    canc_g = cancelled.groupby(["businessId", "year_month"]).agg(
        cancellations=("businessId", "size"),
    ).reset_index()

    edit_g = edited.groupby(["businessId", "year_month"]).agg(
        edits=("businessId", "size"),
    ).reset_index()

    monthly = conf_g.merge(canc_g, on=["businessId", "year_month"], how="outer")
    monthly = monthly.merge(edit_g, on=["businessId", "year_month"], how="outer")
    for col in ["reservations", "covers", "cancellations", "edits"]:
        if col not in monthly.columns:
            monthly[col] = 0.0
        monthly[col] = monthly[col].fillna(0)
    monthly = monthly.rename(columns={"businessId": "restaurantId"})

    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")

    return monthly


def build_platform(events: pd.DataFrame) -> pd.DataFrame:
    """Long-format counts per (restaurant, month, platform) -> spec 6.3.

    Only ReservationConfirm events carry a `service` (platform) value.
    """
    events = events.copy()
    events["businessId"] = events["businessId"].astype(str)
    events["createdAt"] = pd.to_datetime(events["createdAt"], errors="coerce", utc=True)
    events["year_month"] = events["createdAt"].dt.tz_convert(None).dt.to_period("M").astype(str)

    confirmed = events[events["type"] == "ReservationConfirm"]
    platform = (
        confirmed.groupby(["businessId", "year_month", "service"])
        .size()
        .reset_index(name="count")
        .rename(columns={"businessId": "restaurantId"})
    )
    return platform


# --- App-side slicing helpers ---------------------------------------------

def platform_kpis(view: pd.DataFrame) -> dict:
    """Headline numbers for the selected slice, correctly weighted."""
    reservations = view["reservations"].sum()
    cancellations = view["cancellations"].sum()
    covers = view["covers"].sum()
    edits = view["edits"].sum()
    return {
        "reservations": int(reservations),
        "covers": int(covers),
        "avg_party_size": covers / reservations if reservations else float("nan"),
        "cancellation_rate": cancellations / reservations if reservations else float("nan"),
        "edits_per_reservation": edits / reservations if reservations else float("nan"),
        "active_restaurants": int(view["restaurantId"].nunique()),
    }


def monthly_totals(view: pd.DataFrame) -> pd.DataFrame:
    """Reservations/covers/cancellations/edits per month, for the trend line."""
    return (
        view.groupby("year_month", as_index=False)[["reservations", "covers", "cancellations", "edits"]]
        .sum()
        .sort_values("year_month")
    )


def platform_distribution(platform_view: pd.DataFrame) -> pd.DataFrame:
    """Reservation counts by platform (service), summed over the selection."""
    return (
        platform_view.groupby("service", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )


def _restaurant_rollup(view: pd.DataFrame) -> pd.DataFrame:
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        reservations=("reservations", "sum"),
        covers=("covers", "sum"),
        cancellations=("cancellations", "sum"),
        edits=("edits", "sum"),
    )
    g["avg_party_size"] = g["covers"] / g["reservations"].replace(0, np.nan)
    g["cancellation_rate"] = g["cancellations"] / g["reservations"].replace(0, np.nan)
    g["edits_per_reservation"] = g["edits"] / g["reservations"].replace(0, np.nan)
    return g


def top_restaurants(view: pd.DataFrame, metric: str = "reservations",
                    n: int = 10, min_reservations: int = 30,
                    ascending: bool = False) -> pd.DataFrame:
    """Top (or, with ascending=True, bottom) N restaurants by a chosen metric.
    Rate metrics require a min volume so a restaurant with 1 reservation and
    0 cancellations doesn't look like a perfect score."""
    g = _restaurant_rollup(view)
    if metric in ("avg_party_size", "cancellation_rate", "edits_per_reservation"):
        g = g[g["reservations"] >= min_reservations]
    display_cols = ["name", "reservations", "covers", "avg_party_size", "cancellation_rate", "edits_per_reservation"]
    return (
        g.sort_values(metric, ascending=ascending)[display_cols]
        .head(n)
        .reset_index(drop=True)
    )


def restaurant_percentiles(view: pd.DataFrame, metric: str = "covers"):
    """Bottom-25% / top-90% restaurant lists for `metric` (spec's percentile
    spread) -- e.g. metric="covers" for total covers."""
    g = _restaurant_rollup(view)
    return percentile_groups(g, metric)