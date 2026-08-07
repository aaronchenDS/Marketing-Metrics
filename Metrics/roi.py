"""ROI Calculator Baseline (spec: ROI Calculator Baseline / notebook 11.15).

This section is different from the others: it doesn't pull anything new
from Mongo. It combines platform metrics that are ALREADY computed by the
conversations/reservations/takeout modules with business assumptions that
aren't observed in the data at all -- staff cost per call, revenue per
cover -- which the spec sheet marks as "industry benchmark (until customer
data available)". app.py exposes those as adjustable sliders so marketing
can try different assumptions; the defaults below are exactly the numbers
from the spec sheet.

Treat every dollar figure here as "value supported", not "value caused" or
"revenue attributed" -- these are estimates from averages and assumptions,
not observed revenue attribution. Say so wherever the numbers are shown.
"""
DEFAULT_ASSUMPTIONS = {
    "staff_cost_per_call_low": 0.50,
    "staff_cost_per_call_high": 2.00,
    "revenue_per_cover_low": 75.00,
    "revenue_per_cover_high": 150.00,
}


def compute_baseline(conv_view, conv_kpis: dict, res_view, res_kpis: dict,
                      take_kpis: dict, assumptions: dict) -> dict:
    """conv_kpis/res_kpis/take_kpis are the platform_kpis() dicts already
    computed by the other modules for whatever the user selected. conv_view /
    res_view are the matching monthly snapshot slices, used here for the
    "average calls/covers per restaurant-month" figures.
    """
    avg_calls_per_restaurant_month = (
        conv_view["conversations"].mean() if not conv_view.empty else float("nan")
    )
    avg_covers_per_restaurant_month = (
        res_view["covers"].mean() if not res_view.empty else float("nan")
    )

    staff_low = assumptions.get("staff_cost_per_call_low", DEFAULT_ASSUMPTIONS["staff_cost_per_call_low"])
    staff_high = assumptions.get("staff_cost_per_call_high", DEFAULT_ASSUMPTIONS["staff_cost_per_call_high"])
    cover_low = assumptions.get("revenue_per_cover_low", DEFAULT_ASSUMPTIONS["revenue_per_cover_low"])
    cover_high = assumptions.get("revenue_per_cover_high", DEFAULT_ASSUMPTIONS["revenue_per_cover_high"])

    return {
        "avg_call_duration_s": conv_kpis.get("avg_duration_s", float("nan")),
        "not_sent_to_host_rate": 1 - conv_kpis.get("sent_to_host_rate", float("nan")),
        "avg_party_size": res_kpis.get("avg_party_size", float("nan")),
        "avg_takeout_order_value": take_kpis.get("avg_order_value", float("nan")),
        "avg_calls_per_restaurant_month": avg_calls_per_restaurant_month,
        "avg_covers_per_restaurant_month": avg_covers_per_restaurant_month,
        "estimated_labor_value_low": avg_calls_per_restaurant_month * staff_low,
        "estimated_labor_value_high": avg_calls_per_restaurant_month * staff_high,
        "estimated_revenue_supported_low": avg_covers_per_restaurant_month * cover_low,
        "estimated_revenue_supported_high": avg_covers_per_restaurant_month * cover_high,
    }
