"""Takeout metrics (spec: takeout funnel -- carts, orders, items, revenue).

Takeout events, in funnel order (see notebook 11.16.5):
- TakeoutCartCreated       -> a cart was started
- TakeoutOrderCreated      -> a cart converted into an order (may carry a
                               monetary total -- the field name is detected
                               in data_prep.py and normalized to `revenue`)
- TakeoutOrderItemOrdered  -> one row per line item on an order (used as a
                               revenue fallback when no order-level total
                               exists, same as the notebook does)

Same additive-snapshot idea as conversations/reservations: build_monthly
stores counts and sums per (restaurant, month); the app derives conversion
rates and averages by summing those pieces over whatever month range or
restaurant subset the user picks.
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


def build_monthly(carts: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame,
                   restaurants: pd.DataFrame) -> pd.DataFrame:
    """One row per (restaurant, month): carts, orders, items, revenue.

    `orders` is expected to already have a `revenue` column (order-level
    total, or 0.0 if none was found) and `items` an `item_revenue` column --
    both normalized by data_prep.py's field-discovery step.
    """
    carts, orders, items = _month(carts), _month(orders), _month(items)

    cart_g = carts.groupby(["businessId", "year_month"]).size().rename("carts").reset_index()
    order_g = orders.groupby(["businessId", "year_month"]).agg(
        orders=("businessId", "size"),
        revenue=("revenue", "sum"),
    ).reset_index()
    item_g = items.groupby(["businessId", "year_month"]).agg(
        items=("businessId", "size"),
        revenue_from_items=("item_revenue", "sum"),
    ).reset_index()

    monthly = cart_g.merge(order_g, on=["businessId", "year_month"], how="outer")
    monthly = monthly.merge(item_g, on=["businessId", "year_month"], how="outer")
    for col in ["carts", "orders", "items", "revenue", "revenue_from_items"]:
        if col not in monthly.columns:
            monthly[col] = 0.0
        monthly[col] = monthly[col].fillna(0)

    # Prefer the order-level total; fall back to summed item prices only
    # where there's no order-level total (mirrors the notebook's fallback).
    monthly["revenue_effective"] = np.where(
        monthly["revenue"] > 0, monthly["revenue"], monthly["revenue_from_items"]
    )

    monthly = monthly.rename(columns={"businessId": "restaurantId"})
    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")
    return monthly


def build_hourly(orders: pd.DataFrame) -> pd.DataFrame:
    """Platform-wide order counts by (month, hour-of-day) -> busiest-hour chart."""
    orders = _month(orders)
    orders["hour"] = orders["createdAt"].dt.hour
    return (
        orders.dropna(subset=["hour"])
        .groupby(["year_month", "hour"])
        .size()
        .reset_index(name="orders")
    )


# --- App-side slicing helpers ----------------------------------------------

def platform_kpis(view: pd.DataFrame) -> dict:
    carts = view["carts"].sum()
    orders = view["orders"].sum()
    items = view["items"].sum()
    revenue = view["revenue_effective"].sum()
    return {
        "carts": int(carts),
        "orders": int(orders),
        "checkout_conversion": orders / carts if carts else float("nan"),
        "items_per_order": items / orders if orders else float("nan"),
        "revenue": revenue,
        "avg_order_value": revenue / orders if orders else float("nan"),
        "active_restaurants": int(view["restaurantId"].nunique()),
    }


def monthly_totals(view: pd.DataFrame) -> pd.DataFrame:
    """Carts/orders/items/revenue per month, for the trend lines."""
    return (
        view.groupby("year_month", as_index=False)[["carts", "orders", "items", "revenue_effective"]]
        .sum()
        .sort_values("year_month")
        .rename(columns={"revenue_effective": "revenue"})
    )


def hourly_distribution(hourly_view: pd.DataFrame) -> pd.DataFrame:
    """Orders by hour of day, summed over the selection."""
    return (
        hourly_view.groupby("hour", as_index=False)["orders"]
        .sum()
        .sort_values("hour")
    )


def _restaurant_rollup(view: pd.DataFrame) -> pd.DataFrame:
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        carts=("carts", "sum"),
        orders=("orders", "sum"),
        items=("items", "sum"),
        revenue=("revenue_effective", "sum"),
    )
    g["checkout_conversion"] = g["orders"] / g["carts"].replace(0, np.nan)
    g["items_per_order"] = g["items"] / g["orders"].replace(0, np.nan)
    g["avg_order_value"] = g["revenue"] / g["orders"].replace(0, np.nan)
    return g


def top_restaurants(view: pd.DataFrame, metric: str = "orders",
                    n: int = 10, min_orders: int = 30) -> pd.DataFrame:
    """Top N restaurants by a chosen metric. Rate/average metrics require a min volume."""
    g = _restaurant_rollup(view)
    if metric in ("checkout_conversion", "items_per_order", "avg_order_value"):
        g = g[g["orders"] >= min_orders]
    display_cols = ["name", "carts", "orders", "items", "revenue", "avg_order_value", "checkout_conversion"]
    return (
        g.sort_values(metric, ascending=False)[display_cols]
        .head(n)
        .reset_index(drop=True)
    )