"""The dashboard UI. Launch with:  streamlit run app.py

Reads only the parquet snapshots (never Mongo), so it loads fast and needs no
database access. Every widget interaction re-runs this file top-to-bottom;
the @st.cache_data in data_access keeps that cheap.
"""
import plotly.express as px
import streamlit as st

import config
from data_access import load_snapshot
from Metrics import conversations as m_conv
from Metrics import reservations as m_res
from Metrics import takeout as m_take
from Metrics import leads as m_leads
from Metrics import tags as m_tags
from Metrics import transfers as m_transfers
from Metrics import roi as m_roi

# Fixed UTC offsets for the "view in timezone" control on the takeout hourly
# chart. These are single platform-wide shifts, not per-restaurant local time
# (restaurants operate worldwide and don't have a timezone field on record --
# see data_prep.py's build_takeout()) and don't adjust for daylight saving.
TIMEZONE_OFFSETS = {
    "UTC": 0,
    "US Eastern (UTC-5)": -5,
    "US Central (UTC-6)": -6,
    "US Mountain (UTC-7)": -7,
    "US Pacific (UTC-8)": -8,
    "UK / London (UTC+0)": 0,
    "Central Europe (UTC+1)": 1,
}

st.set_page_config(page_title="Hostie — Platform Metrics", layout="wide")
st.title("Hostie — Platform Metrics")
st.caption(
    "After-hours metrics and lead-to-booking conversion are not shown yet -- "
    "both are blocked on data gaps (restaurant timezone, and confirming which "
    "event marks a lead as booked) noted in the notebook's own limitations section."
)

# --- Load the snapshot -----------------------------------------------------
monthly = load_snapshot("conversations_monthly")
if monthly.empty:
    st.warning("No snapshot found. Run `python data_prep.py` first to build it.")
    st.stop()

# --- Sidebar: the interactive controls ------------------------------------
st.sidebar.header("Filters")
months = sorted(m for m in monthly["year_month"].dropna().unique())

if len(months) > 1:
    lo, hi = st.sidebar.select_slider(
        "Month range", options=months, value=(months[0], months[-1])
    )
else:
    lo, hi = months[0], months[0]

top_n = st.sidebar.slider("Top N restaurants", 5, 25, 10)
min_conv = st.sidebar.number_input(
    "Min conversations for rate rankings", 0, 2000, config.MIN_CONVERSATIONS, step=5
)
rank_metric = st.sidebar.selectbox(
    "Rank restaurants by",
    ["conversations", "sent_to_host_rate", "avg_score", "avg_duration_s"],
)

view = monthly[(monthly["year_month"] >= lo) & (monthly["year_month"] <= hi)]

# --- KPI row ---------------------------------------------------------------
k = m_conv.platform_kpis(view)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total conversations", f"{k['conversations']:,}")
c2.metric("Active restaurants", f"{k['active_restaurants']:,}")
c3.metric("Avg call duration", f"{k['avg_duration_s']:.0f}s")
c4.metric("Total time spent", f"{k['total_time_hours']:,.0f}h")
c5.metric("Sent-to-host rate", f"{k['sent_to_host_rate']:.1%}")
c6.metric("Avg conversation score", f"{k['avg_score']:.2f}")

# --- Trend -----------------------------------------------------------------
st.subheader("Conversations by month")
trend = m_conv.monthly_totals(view)
fig = px.line(trend, x="year_month", y="conversations", markers=True)
fig.update_traces(line_color=config.PALETTE["primary"])
fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

# --- Percentile spread (spec: "Customers in bottom 25%" / "top 90%") -------
with st.expander("Percentile spread: total conversations & resolved-by-Hostie rate"):
    bottom_conv, top_conv = m_conv.restaurant_percentiles(view, "conversations")
    st.caption("Bottom 25% by total conversations")
    st.dataframe(bottom_conv[["name", "conversations"]], use_container_width=True)
    st.caption("At or above the 90th percentile by total conversations")
    st.dataframe(top_conv[["name", "conversations"]], use_container_width=True)

    bottom_host, top_host = m_conv.restaurant_percentiles(view, "sent_to_host_rate")
    st.caption("Bottom 25% by resolved-by-Hostie rate")
    st.dataframe(bottom_host[["name", "sent_to_host_rate"]], use_container_width=True)
    st.caption("At or above the 90th percentile by resolved-by-Hostie rate")
    st.dataframe(top_host[["name", "sent_to_host_rate"]], use_container_width=True)

# --- Ranking table ---------------------------------------------------------
st.subheader(f"Top {top_n} restaurants by {rank_metric.replace('_', ' ')}")
st.dataframe(
    m_conv.top_restaurants(view, rank_metric, top_n, min_conv),
    use_container_width=True,
)

# ============================================================================
# New vs. Repeat Callers (spec section, notebook 11.4)
# ============================================================================
st.divider()
st.header("New vs. Repeat Callers")

callers_monthly = load_snapshot("conversations_callers_monthly")
if callers_monthly.empty:
    st.info(
        "No caller snapshot found yet. This needs a `phoneFrom` field on "
        "conversations -- run `python data_prep.py` and check its printed "
        "output; if it says the field wasn't found, this section will stay empty."
    )
else:
    callers_view = callers_monthly[(callers_monthly["year_month"] >= lo) & (callers_monthly["year_month"] <= hi)]
    ck = m_conv.caller_kpis(callers_view)
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total new callers", f"{ck['new_callers']:,}")
    n2.metric("Total repeat callers", f"{ck['repeat_callers']:,}")
    n3.metric("New vs. repeat split", f"{ck['new_caller_share']:.1%} / {ck['repeat_caller_share']:.1%}")
    n4.metric("Avg calls/repeat caller/mo", f"{ck['avg_calls_per_repeat_caller_month']:.2f}")

    st.subheader("New caller trend (MoM growth)")
    caller_trend = m_conv.new_caller_trend(callers_view)
    fig_nc = px.bar(caller_trend, x="year_month", y="new_callers")
    fig_nc.update_traces(marker_color=config.PALETTE["primary"])
    fig_nc.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="new callers")
    st.plotly_chart(fig_nc, use_container_width=True)
    st.caption("Month-over-month growth in new callers:")
    st.dataframe(
        caller_trend[["year_month", "new_callers", "mom_growth"]],
        use_container_width=True,
    )

# ============================================================================
# Simultaneous Calls (spec section, notebook 11.6)
# ============================================================================
st.divider()
st.header("Simultaneous Calls")

sim_monthly = load_snapshot("conversations_simultaneous_monthly")
if sim_monthly.empty:
    st.info("No simultaneous-calls snapshot found yet. Run `python data_prep.py` to build it.")
else:
    sim_view = sim_monthly[(sim_monthly["year_month"] >= lo) & (sim_monthly["year_month"] <= hi)]
    sk = m_conv.simultaneous_kpis(sim_view)
    s1, s2 = st.columns(2)
    s1.metric("Total calls", f"{sk['total_calls']:,}")
    s2.metric("Simultaneous-call rate", f"{sk['simultaneous_call_rate']:.2%}")

    with st.expander("Percentile spread: simultaneous-call rate"):
        bottom_sim, top_sim = m_conv.simultaneous_percentiles(sim_view)
        st.caption("Bottom 25% by simultaneous-call rate")
        st.dataframe(bottom_sim[["name", "simultaneous_call_rate"]], use_container_width=True)
        st.caption("At or above the 90th percentile by simultaneous-call rate")
        st.dataframe(top_sim[["name", "simultaneous_call_rate"]], use_container_width=True)

# ============================================================================
# Unanswered Transfer Rate (spec section, notebook 11.7)
# ============================================================================
st.divider()
st.header("Unanswered Transfer Rate")

transfers_monthly = load_snapshot("transfers_monthly")
if transfers_monthly.empty:
    st.info("No transfers snapshot found yet. Run `python data_prep.py` to build it.")
else:
    transfers_view = transfers_monthly[(transfers_monthly["year_month"] >= lo) & (transfers_monthly["year_month"] <= hi)]
    trk = m_transfers.platform_kpis(transfers_view)
    tr1, tr2 = st.columns(2)
    tr1.metric("Transfer attempts", f"{trk['transfer_attempts']:,}")
    tr2.metric("Unanswered transfer rate", f"{trk['unanswered_transfer_rate']:.1%}")

    st.subheader("Restaurants flagged above 25% unanswered")
    st.dataframe(
        m_transfers.flagged_restaurants(transfers_view, threshold=0.25),
        use_container_width=True,
    )

# ============================================================================
# Reservations section (spec 6-8: reservations, covers, cancellations, edits)
# ============================================================================
st.divider()
st.header("Reservations")

res_monthly = load_snapshot("reservations_monthly")
res_platform = load_snapshot("reservations_platform")

if res_monthly.empty:
    st.info("No reservations snapshot found yet. Run `python data_prep.py` to build it.")
else:
    res_view = res_monthly[(res_monthly["year_month"] >= lo) & (res_monthly["year_month"] <= hi)]
    platform_view = res_platform[(res_platform["year_month"] >= lo) & (res_platform["year_month"] <= hi)]

    rk = m_res.platform_kpis(res_view)
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Total reservations", f"{rk['reservations']:,}")
    r2.metric("Total covers", f"{rk['covers']:,}")
    r3.metric("Avg party size", f"{rk['avg_party_size']:.1f}")
    r4.metric("Cancellation rate", f"{rk['cancellation_rate']:.1%}")
    r5.metric("Edits per reservation", f"{rk['edits_per_reservation']:.2f}")
    r6.metric("Active restaurants", f"{rk['active_restaurants']:,}")

    with st.expander("Percentile spread: total covers"):
        bottom_cov, top_cov = m_res.restaurant_percentiles(res_view, "covers")
        st.caption("Bottom 25% by total covers")
        st.dataframe(bottom_cov[["name", "covers"]], use_container_width=True)
        st.caption("At or above the 90th percentile by total covers")
        st.dataframe(top_cov[["name", "covers"]], use_container_width=True)

    st.sidebar.header("Reservation filters")
    res_min = st.sidebar.number_input(
        "Min reservations for rate rankings", 0, 2000, config.MIN_CONVERSATIONS, step=5
    )
    res_rank_metric = st.sidebar.selectbox(
        "Rank restaurants by (reservations)",
        ["reservations", "covers", "avg_party_size", "cancellation_rate", "edits_per_reservation"],
    )

    st.subheader("Reservations by month")
    res_trend = m_res.monthly_totals(res_view)
    fig2 = px.line(res_trend, x="year_month", y="reservations", markers=True)
    fig2.update_traces(line_color=config.PALETTE["primary"])
    fig2.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Reservation platform distribution")
    dist = m_res.platform_distribution(platform_view)
    fig3 = px.bar(dist, x="service", y="count")
    fig3.update_traces(marker_color=config.PALETTE["accent"])
    fig3.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader(f"Top {top_n} restaurants by {res_rank_metric.replace('_', ' ')} (reservations)")
    st.dataframe(
        m_res.top_restaurants(res_view, res_rank_metric, top_n, res_min),
        use_container_width=True,
    )

    st.subheader(f"Top {top_n} restaurants by LOWEST cancellation rate")
    st.dataframe(
        m_res.top_restaurants(res_view, "cancellation_rate", top_n, res_min, ascending=True),
        use_container_width=True,
    )

# ============================================================================
# Takeout section (spec: takeout funnel -- carts, orders, items, revenue)
# ============================================================================
st.divider()
st.header("Takeout")

take_monthly = load_snapshot("takeout_monthly")
take_hourly = load_snapshot("takeout_hourly")

if take_monthly.empty:
    st.info("No takeout snapshot found yet. Run `python data_prep.py` to build it.")
else:
    take_view = take_monthly[(take_monthly["year_month"] >= lo) & (take_monthly["year_month"] <= hi)]
    hourly_view = take_hourly[(take_hourly["year_month"] >= lo) & (take_hourly["year_month"] <= hi)]

    tk = m_take.platform_kpis(take_view)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Carts started", f"{tk['carts']:,}")
    t2.metric("Orders completed", f"{tk['orders']:,}")
    t3.metric("Checkout conversion", f"{tk['checkout_conversion']:.1%}")
    t4.metric("Items per order", f"{tk['items_per_order']:.2f}")
    t5, t6, t7 = st.columns(3)
    t5.metric("Total revenue", f"${tk['revenue']:,.2f}")
    t6.metric("Avg order value", f"${tk['avg_order_value']:,.2f}")
    t7.metric("Active restaurants", f"{tk['active_restaurants']:,}")

    st.sidebar.header("Takeout filters")
    take_min = st.sidebar.number_input(
        "Min orders for rate rankings", 0, 2000, config.MIN_CONVERSATIONS, step=5
    )
    take_rank_metric = st.sidebar.selectbox(
        "Rank restaurants by (takeout)",
        ["orders", "revenue", "checkout_conversion", "items_per_order", "avg_order_value"],
    )

    st.subheader("Takeout orders & revenue by month")
    take_trend = m_take.monthly_totals(take_view)
    fig4 = px.line(take_trend, x="year_month", y="orders", markers=True)
    fig4.update_traces(line_color=config.PALETTE["primary"])
    fig4.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="orders")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.line(take_trend, x="year_month", y="revenue", markers=True)
    fig5.update_traces(line_color=config.PALETTE["accent"])
    fig5.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="revenue ($)")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Orders by hour of day")
    tz_label = st.selectbox("View in timezone", list(TIMEZONE_OFFSETS.keys()), key="takeout_tz")
    st.caption(
        "Restaurants operate worldwide and don't have a timezone field on record, "
        "so this shifts every order by one fixed offset for the zone you pick -- "
        "it won't be exact for restaurants outside that zone, and doesn't adjust "
        "for daylight saving. Pick UTC to see the raw, unshifted data."
    )
    hourly_dist = m_take.hourly_distribution(hourly_view)
    offset = TIMEZONE_OFFSETS[tz_label]
    hourly_dist["hour"] = (hourly_dist["hour"] + offset) % 24
    hourly_dist = hourly_dist.sort_values("hour")
    fig6 = px.bar(hourly_dist, x="hour", y="orders")
    fig6.update_traces(marker_color=config.PALETTE["warn"])
    fig6.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title=f"hour ({tz_label})", yaxis_title="")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader(f"Top {top_n} restaurants by {take_rank_metric.replace('_', ' ')} (takeout)")
    st.dataframe(
        m_take.top_restaurants(take_view, take_rank_metric, top_n, take_min),
        use_container_width=True,
    )

# ============================================================================
# Events / Leads section (spec: Events -- EventBookingLeadCreated)
# ============================================================================
st.divider()
st.header("Events / Leads")

leads_monthly = load_snapshot("leads_monthly")
leads_platform = load_snapshot("leads_platform")

if leads_monthly.empty:
    st.info("No leads snapshot found yet. Run `python data_prep.py` to build it.")
else:
    leads_view = leads_monthly[(leads_monthly["year_month"] >= lo) & (leads_monthly["year_month"] <= hi)]
    leads_platform_view = leads_platform[(leads_platform["year_month"] >= lo) & (leads_platform["year_month"] <= hi)]

    lk = m_leads.platform_kpis(leads_view)
    l1, l2, l3 = st.columns(3)
    l1.metric("Leads captured", f"{lk['leads_captured']:,}")
    l2.metric("Avg leads per restaurant", f"{lk['avg_leads_per_restaurant']:.2f}")
    avg_size = lk["avg_event_size"]
    l3.metric("Avg event size", f"{avg_size:.1f}" if avg_size == avg_size else "n/a")

    st.caption(
        "Lead-to-booking conversion isn't shown yet -- pending confirmation of "
        "which event marks a lead as booked (PerfectVenueCreate is a candidate, "
        "unconfirmed per the notebook)."
    )

    st.sidebar.header("Leads filters")
    leads_min = st.sidebar.number_input("Min leads for rate rankings", 0, 500, 5, step=1)
    leads_rank_metric = st.sidebar.selectbox(
        "Rank restaurants by (leads)", ["leads_captured", "avg_event_size"]
    )

    st.subheader("Leads captured by month")
    leads_trend = m_leads.monthly_totals(leads_view)
    fig7 = px.line(leads_trend, x="year_month", y="leads_captured", markers=True)
    fig7.update_traces(line_color=config.PALETTE["primary"])
    fig7.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Leads by platform")
    leads_dist = m_leads.platform_distribution(leads_platform_view)
    fig8 = px.bar(leads_dist, x="platform", y="count")
    fig8.update_traces(marker_color=config.PALETTE["accent"])
    fig8.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader(f"Top {top_n} restaurants by {leads_rank_metric.replace('_', ' ')} (leads)")
    st.dataframe(
        m_leads.top_restaurants(leads_view, leads_rank_metric, top_n, leads_min),
        use_container_width=True,
    )

# ============================================================================
# Conversation Tags section (spec: Conversation Tags -- sections 10/11.11/11.16.2)
# ============================================================================
st.divider()
st.header("Conversation Tags")

tags_monthly = load_snapshot("tags_monthly")
tags_by_channel = load_snapshot("tags_by_channel")

if tags_monthly.empty:
    st.info("No tags snapshot found yet. Run `python data_prep.py` to build it.")
else:
    tags_view = tags_monthly[(tags_monthly["year_month"] >= lo) & (tags_monthly["year_month"] <= hi)]
    channel_view = tags_by_channel[(tags_by_channel["year_month"] >= lo) & (tags_by_channel["year_month"] <= hi)]

    tgk = m_tags.platform_kpis(tags_view)
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Conversations", f"{tgk['conversations']:,}")
    g2.metric("Speak-to-human rate", f"{tgk['speak_human_rate']:.1%}")
    g3.metric("Miscommunication rate", f"{tgk['miscommunication_rate']:.1%}")
    g4.metric("Frustrated rate", f"{tgk['frustrated_rate']:.1%}")
    g5.metric("Cancellation tag rate", f"{tgk['cancellation_rate']:.1%}")

    st.sidebar.header("Tags filters")
    tags_min = st.sidebar.number_input(
        "Min conversations for rate rankings (tags)", 0, 2000, config.MIN_CONVERSATIONS, step=5
    )
    tags_rank_metric = st.sidebar.selectbox(
        "Rank restaurants by (tags)",
        ["conversations", "speak_human_rate", "miscommunication_rate", "frustrated_rate", "cancellation_rate"],
    )

    st.subheader("Top tags by volume (platform-wide)")
    st.dataframe(m_tags.top_tags_overall(channel_view, top_n), use_container_width=True)

    st.subheader("Top tags by channel (call vs. text)")
    st.dataframe(
        m_tags.top_tags_by_channel(channel_view, top_n),
        use_container_width=True,
    )

    st.subheader("Tag rates by month")
    monthly_rates = m_tags.monthly_tag_rates(tags_view)
    fig9 = px.line(
        monthly_rates, x="year_month",
        y=["speak_human_rate", "miscommunication_rate", "frustrated_rate", "cancellation_rate"],
        markers=True,
    )
    fig9.update_layout(margin=dict(t=10, b=0, l=0, r=0), xaxis_title="", yaxis_title="rate", legend_title="")
    st.plotly_chart(fig9, use_container_width=True)

    with st.expander("Percentile spread: speak-to-human rate"):
        bottom_sh, top_sh = m_tags.restaurant_percentiles(tags_view, "speak_human_rate")
        st.caption("Bottom 25% by speak-to-human rate")
        st.dataframe(bottom_sh[["name", "speak_human_rate"]], use_container_width=True)
        st.caption("At or above the 90th percentile by speak-to-human rate")
        st.dataframe(top_sh[["name", "speak_human_rate"]], use_container_width=True)

    st.subheader("Frustrated-rate CS outliers")
    st.caption(
        "Restaurants whose frustrated-tag rate is more than 2 standard deviations "
        "above the platform mean, among restaurants with enough volume to be meaningful."
    )
    st.dataframe(
        m_tags.frustrated_outliers(tags_view, tags_min),
        use_container_width=True,
    )

    st.subheader(f"Top {top_n} restaurants by {tags_rank_metric.replace('_', ' ')} (tags)")
    st.dataframe(
        m_tags.top_restaurants(tags_view, tags_rank_metric, top_n, tags_min),
        use_container_width=True,
    )

# ============================================================================
# ROI Calculator Baseline (spec / notebook 11.15)
# ============================================================================
st.divider()
st.header("ROI Calculator Baseline")
st.caption(
    "Combines real platform metrics with business assumptions that aren't "
    "observed in the data (staff cost/call, revenue/cover -- industry "
    "benchmarks until real customer data is available). Treat every dollar "
    "figure below as value SUPPORTED, not value CAUSED, unless attribution "
    "is separately validated."
)

if res_monthly.empty or take_monthly.empty:
    st.info("Needs the Reservations and Takeout snapshots above to be built first.")
else:
    st.sidebar.header("ROI assumptions")
    staff_low = st.sidebar.number_input("Staff cost per call, low ($)", 0.0, 50.0, 0.50, step=0.10)
    staff_high = st.sidebar.number_input("Staff cost per call, high ($)", 0.0, 50.0, 2.00, step=0.10)
    cover_low = st.sidebar.number_input("Revenue per cover, low ($)", 0.0, 1000.0, 75.00, step=5.0)
    cover_high = st.sidebar.number_input("Revenue per cover, high ($)", 0.0, 1000.0, 150.00, step=5.0)

    assumptions = {
        "staff_cost_per_call_low": staff_low,
        "staff_cost_per_call_high": staff_high,
        "revenue_per_cover_low": cover_low,
        "revenue_per_cover_high": cover_high,
    }
    baseline = m_roi.compute_baseline(view, k, res_view, rk, tk, assumptions)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Avg call duration", f"{baseline['avg_call_duration_s']:.0f}s")
    b2.metric("Not-sent-to-host rate", f"{baseline['not_sent_to_host_rate']:.1%}")
    b3.metric("Avg party size", f"{baseline['avg_party_size']:.2f}")
    b4.metric("Avg takeout order value", f"${baseline['avg_takeout_order_value']:,.2f}")

    b5, b6 = st.columns(2)
    b5.metric("Avg calls / restaurant-month", f"{baseline['avg_calls_per_restaurant_month']:,.0f}")
    b6.metric("Avg covers / restaurant-month", f"{baseline['avg_covers_per_restaurant_month']:,.0f}")

    st.subheader("Estimated monthly value supported per restaurant")
    st.metric(
        "Labor value (from call volume)",
        f"${baseline['estimated_labor_value_low']:,.2f} – ${baseline['estimated_labor_value_high']:,.2f}",
    )
    st.metric(
        "Revenue supported (from covers)",
        f"${baseline['estimated_revenue_supported_low']:,.2f} – ${baseline['estimated_revenue_supported_high']:,.2f}",
    )