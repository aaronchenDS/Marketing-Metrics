"""Leads metrics (spec: Events -- leads captured via EventBookingLeadCreated).

Leads come from `events` where type == "EventBookingLeadCreated" (notebook
11.16.4). The platform/source field and event-size field aren't guaranteed
to have fixed names, so data_prep.py auto-detects them from a real document
sample (same idea as the takeout money-field detection) and normalizes them
here to `platform` / `event_size`.

Lead-to-booking conversion is intentionally NOT computed here -- the
notebook flags this as pending confirmation of which event marks a lead as
"booked" (PerfectVenueCreate is a candidate, unconfirmed). Once that's
confirmed, add a booked-events pull and a conversion rate the same way this
module already tracks leads captured.
"""
import numpy as np
import pandas as pd


def _month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "businessId" not in df.columns:
        df["businessId"] = pd.Series(dtype=str)
    if "createdAt" not in df.columns:
        df["createdAt"] = pd.Series(dtype="datetime64[ns]")
    df["businessId"] = df["businessId"].astype(str)
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    df["year_month"] = df["createdAt"].dt.tz_convert(None).dt.to_period("M").astype(str)
    return df


def build_monthly(leads: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """One row per (restaurant, month): leads captured + event-size sum/count."""
    leads = _month(leads)
    if "event_size" not in leads.columns:
        leads["event_size"] = np.nan
    leads["event_size"] = pd.to_numeric(leads["event_size"], errors="coerce")

    monthly = leads.groupby(["businessId", "year_month"]).agg(
        leads_captured=("businessId", "size"),
        size_sum=("event_size", "sum"),
        size_n=("event_size", "count"),
    ).reset_index()

    monthly = monthly.rename(columns={"businessId": "restaurantId"})
    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")
    return monthly


def build_platform(leads: pd.DataFrame) -> pd.DataFrame:
    """Long-format lead counts per (month, platform) -- platform-wide, not per restaurant."""
    leads = _month(leads)
    if "platform" not in leads.columns:
        leads["platform"] = "Unknown"
    leads["platform"] = leads["platform"].fillna("Unknown")
    return (
        leads.groupby(["year_month", "platform"])
        .size()
        .reset_index(name="count")
    )


# --- App-side slicing helpers ----------------------------------------------

def platform_kpis(view: pd.DataFrame) -> dict:
    leads = view["leads_captured"].sum()
    size_n = view["size_n"].sum()
    restaurants = view["restaurantId"].nunique()
    return {
        "leads_captured": int(leads),
        "active_restaurants": int(restaurants),
        "avg_leads_per_restaurant": leads / restaurants if restaurants else float("nan"),
        "avg_event_size": view["size_sum"].sum() / size_n if size_n else float("nan"),
    }


def monthly_totals(view: pd.DataFrame) -> pd.DataFrame:
    """Leads captured per month, for the trend line."""
    return (
        view.groupby("year_month", as_index=False)["leads_captured"]
        .sum()
        .sort_values("year_month")
    )


def platform_distribution(platform_view: pd.DataFrame) -> pd.DataFrame:
    """Lead counts by platform, summed over the selection."""
    return (
        platform_view.groupby("platform", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )


def _restaurant_rollup(view: pd.DataFrame) -> pd.DataFrame:
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        leads_captured=("leads_captured", "sum"),
        size_sum=("size_sum", "sum"),
        size_n=("size_n", "sum"),
    )
    g["avg_event_size"] = g["size_sum"] / g["size_n"].replace(0, np.nan)
    return g


def top_restaurants(view: pd.DataFrame, metric: str = "leads_captured",
                    n: int = 10, min_leads: int = 5) -> pd.DataFrame:
    """Top N restaurants by a chosen metric. avg_event_size requires a min volume."""
    g = _restaurant_rollup(view)
    if metric == "avg_event_size":
        g = g[g["leads_captured"] >= min_leads]
    display_cols = ["name", "leads_captured", "avg_event_size"]
    return (
        g.sort_values(metric, ascending=False)[display_cols]
        .head(n)
        .reset_index(drop=True)
    )