"""ETL: pull from MongoDB, compute snapshots, write parquet. Run monthly:

    python data_prep.py

Only this file talks to Mongo. It reuses the notebook's own helpers (`load`,
`clean_object_id`, `convo_fields`) so the logic matches what you already tested.
"""
import pandas as pd
from pymongo import MongoClient

import config
from Metrics import conversations as m_conv
from Metrics import reservations as m_res
from Metrics import takeout as m_take
from Metrics import leads as m_leads
from Metrics import tags as m_tags
from Metrics import transfers as m_transfers

client = MongoClient(config.MONGO_URL)
db = client[config.DB_NAME]


# --- notebook helpers, verbatim -------------------------------------------
def load(collection, query=None, fields=None):
    return pd.DataFrame(list(db[collection].find(query or {}, fields)))


def clean_object_id(series):
    return (
        series.astype(str)
        .str.replace("ObjectId(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )


def drop_epoch_dates(df, date_col, label):
    """Drop rows whose `date_col` falls in 1970 -- the classic sentinel for a
    missing/unset timestamp (Mongo/JS `new Date(0)` or `Date(null)` both land
    on 1970-01-01), not a real historical record. Applied right after each
    raw pull, before anything downstream buckets rows by month.
    """
    if date_col not in df.columns:
        return df
    parsed = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    is_1970 = parsed.dt.year == 1970
    n_dropped = int(is_1970.sum())
    if n_dropped:
        print(f"  dropping {n_dropped:,} {label} rows with a 1970 {date_col} (bad/missing timestamp)")
    return df[~is_1970]


# Restaurants to scrub from every snapshot, platform-wide: anything actually
# listed in `demo_restaurants`, PLUS any restaurant whose name merely
# contains "demo" or "test" in any casing (demo/Demo/DEMO, test/Test/TEST) --
# catches internal QA/demo accounts that were never added to that
# collection. Computed once in main() and threaded into every build_*
# function below so all six snapshots apply the exact same exclusion list.
NAME_EXCLUDE_PATTERN = "demo|test|onboarding|trial"


def load_excluded_restaurant_ids():
    demo_ids = set(clean_object_id(pd.Series(db["demo_restaurants"].distinct("_id"))))

    restaurants = load("restaurants", fields={"_id": 1, "name": 1})
    if restaurants.empty:
        name_ids = set()
    else:
        name_flag = restaurants["name"].astype(str).str.contains(
            NAME_EXCLUDE_PATTERN, case=False, na=False, regex=True
        )
        name_ids = set(clean_object_id(restaurants.loc[name_flag, "_id"]))

    excluded = demo_ids | name_ids
    print(
        f"  excluding {len(excluded):,} restaurants "
        f"({len(demo_ids):,} from demo_restaurants, {len(name_ids):,} by name match)"
    )
    return excluded


# Same projection idea as the notebook's convo_fields, trimmed to what the
# conversations/callers/simultaneous snapshots need. `phoneFrom` is needed
# for new-vs-repeat callers (notebook 11.4).
convo_fields = {
    "_id": 0,
    "restaurantId": 1,
    "dateCreated": 1,
    "dateEnded": 1,
    "convoSentToHost": 1,
    "conversationScore": 1,
    "phoneFrom": 1,
}


def build_conversations(excluded_ids):
    """Conversations section -> conversations_monthly.parquet,
    conversations_callers_monthly.parquet, conversations_simultaneous_monthly.parquet"""
    if config.DEV_LIMIT:
        # random representative sample -> includes recent (scored) rows,
        # not just the oldest DEV_LIMIT docs in insertion order
        cursor = db["conversations"].aggregate(
            [{"$sample": {"size": config.DEV_LIMIT}}, {"$project": convo_fields}],
            allowDiskUse=True,
        )
    else:
        cursor = db["conversations"].find({}, convo_fields)
    conv = pd.DataFrame(list(cursor))
    print(f"  pulled {len(conv):,} conversations")

    conv = drop_epoch_dates(conv, "dateCreated", "conversations")
    conv = conv[~conv["restaurantId"].astype(str).isin(excluded_ids)]
    restaurants = load("restaurants", fields={"_id": 1, "name": 1})

    monthly = m_conv.build_monthly(conv, restaurants)
    out = config.Data_Directory / "conversations_monthly.parquet"
    monthly.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(monthly):,} restaurant-months)")

    if "phoneFrom" in conv.columns:
        callers = m_conv.build_callers_monthly(conv, restaurants)
        out_c = config.Data_Directory / "conversations_callers_monthly.parquet"
        callers.to_parquet(out_c, index=False)
        print(f"  wrote {out_c}  ({len(callers):,} restaurant-months)")
    else:
        print("  'phoneFrom' not found on conversations -- skipping new-vs-repeat callers")

    simultaneous = m_conv.build_simultaneous_monthly(conv, restaurants)
    out_s = config.Data_Directory / "conversations_simultaneous_monthly.parquet"
    simultaneous.to_parquet(out_s, index=False)
    print(f"  wrote {out_s}  ({len(simultaneous):,} restaurant-months)")


# Reservations live in `events`, not a `reservations` collection -- see
# notebook section 6. ReservationConfirm = a completed reservation (has
# partySize/service); ReservationCancel = a cancellation; ReservationEdit =
# an edit (deliberately excludes the distinct "ReservationEditFailed" event,
# per notebook 11.16.1 -- confirmed exact event names via Compass).
reservation_event_fields = {
    "_id": 0,
    "type": 1,
    "businessId": 1,
    "createdAt": 1,
    "partySize": 1,
    "service": 1,
}


def build_reservations(excluded_ids):
    """Reservations/covers/cancellations/edits -> reservations_monthly.parquet
    and reservations_platform.parquet"""
    match = {"type": {"$in": ["ReservationConfirm", "ReservationCancel", "ReservationEdit"]}}
    if config.DEV_LIMIT:
        cursor = db["events"].aggregate(
            [
                {"$match": match},
                {"$sample": {"size": config.DEV_LIMIT}},
                {"$project": reservation_event_fields},
            ],
            allowDiskUse=True,
        )
    else:
        cursor = db["events"].find(match, reservation_event_fields)
    events = pd.DataFrame(list(cursor))
    print(f"  pulled {len(events):,} reservation events")

    events = drop_epoch_dates(events, "createdAt", "reservation events")
    events = events[~events["businessId"].astype(str).isin(excluded_ids)]
    restaurants = load("restaurants", fields={"_id": 1, "name": 1})

    monthly = m_res.build_monthly(events, restaurants)
    out = config.Data_Directory / "reservations_monthly.parquet"
    monthly.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(monthly):,} restaurant-months)")

    platform = m_res.build_platform(events)
    out2 = config.Data_Directory / "reservations_platform.parquet"
    platform.to_parquet(out2, index=False)
    print(f"  wrote {out2}  ({len(platform):,} rows)")


# Takeout funnel: TakeoutCartCreated -> TakeoutOrderCreated ->
# TakeoutOrderItemOrdered (see notebook 11.16.5). The monetary field isn't
# guaranteed to have the same name on every deployment, so we sample a few
# real documents first and detect it -- same idea as the notebook's
# "field discovery" cell (11.16.5's first cell).
# Confirmed via Compass: TakeoutOrderCreated stores its total as `totalCents`
# (Int32, cents) -- listed first so it wins over the plainer "total" guess.
ORDER_MONEY_CANDIDATES = ["totalCents", "total", "totalAmount", "amount", "price", "subtotal", "grandTotal", "revenue"]
ITEM_MONEY_CANDIDATES = ["priceCents", "price", "total", "amount", "unitPrice"]


def _detect_money_field(event_type, candidates):
    """Look at a few real documents of this event type and find a money field."""
    sample = list(db["events"].find({"type": event_type}).limit(5))
    if not sample:
        return None
    cols = set()
    for doc in sample:
        cols.update(doc.keys())
    for c in candidates:
        if c in cols:
            return c
    # Fallback: anything that merely looks money-shaped. Sorted so the choice
    # is deterministic run-to-run rather than depending on set iteration order.
    for c in sorted(cols):
        if any(term in c.lower() for term in ["total", "amount", "price", "revenue", "cent"]):
            return c
    return None


def _is_cents_field(field_name: str) -> bool:
    """Field names like totalCents / priceCents store integer cents, not dollars."""
    return bool(field_name) and "cent" in field_name.lower()


def _pull_events(evtype, excluded_ids, extra_field=None):
    """Sample (or fully pull) one event type, projecting only what's needed."""
    fields = {"_id": 0, "businessId": 1, "createdAt": 1}
    if extra_field:
        fields[extra_field] = 1
    match = {"type": evtype}
    if config.DEV_LIMIT:
        cursor = db["events"].aggregate(
            [{"$match": match}, {"$sample": {"size": config.DEV_LIMIT}}, {"$project": fields}],
            allowDiskUse=True,
        )
    else:
        cursor = db["events"].find(match, fields)
    out = pd.DataFrame(list(cursor))
    out = drop_epoch_dates(out, "createdAt", evtype)
    if "businessId" in out.columns:
        out = out[~out["businessId"].astype(str).isin(excluded_ids)]
    return out


def build_takeout(excluded_ids):
    """Takeout funnel -> takeout_monthly.parquet + takeout_hourly.parquet"""
    order_money_field = _detect_money_field("TakeoutOrderCreated", ORDER_MONEY_CANDIDATES)
    item_money_field = _detect_money_field("TakeoutOrderItemOrdered", ITEM_MONEY_CANDIDATES)
    print(f"  order money field: {order_money_field!r}, item money field: {item_money_field!r}")

    if order_money_field:
        _sample_docs = list(db["events"].find({"type": "TakeoutOrderCreated"}).limit(8))
        _raw_values = [d.get(order_money_field) for d in _sample_docs]
        print(f"  sample raw {order_money_field!r} values: {_raw_values}")

    carts = _pull_events("TakeoutCartCreated", excluded_ids)
    orders = _pull_events("TakeoutOrderCreated", excluded_ids, order_money_field)
    items = _pull_events("TakeoutOrderItemOrdered", excluded_ids, item_money_field)
    print(f"  pulled {len(carts):,} carts, {len(orders):,} orders, {len(items):,} items")

    if order_money_field and order_money_field in orders.columns:
        orders = orders.rename(columns={order_money_field: "revenue"})
        orders["revenue"] = pd.to_numeric(orders["revenue"], errors="coerce")
        if _is_cents_field(order_money_field):
            orders["revenue"] = orders["revenue"] / 100
            print(f"  '{order_money_field}' is cents -> divided by 100 to get dollars")
    else:
        orders["revenue"] = 0.0

    if item_money_field and item_money_field in items.columns:
        items = items.rename(columns={item_money_field: "item_revenue"})
        items["item_revenue"] = pd.to_numeric(items["item_revenue"], errors="coerce")
        if _is_cents_field(item_money_field):
            items["item_revenue"] = items["item_revenue"] / 100
            print(f"  '{item_money_field}' is cents -> divided by 100 to get dollars")
    else:
        items["item_revenue"] = 0.0

    restaurants = load("restaurants", fields={"_id": 1, "name": 1})

    monthly = m_take.build_monthly(carts, orders, items, restaurants)
    out = config.Data_Directory / "takeout_monthly.parquet"
    monthly.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(monthly):,} restaurant-months)")

    hourly = m_take.build_hourly(orders)
    out2 = config.Data_Directory / "takeout_hourly.parquet"
    hourly.to_parquet(out2, index=False)
    print(f"  wrote {out2}  ({len(hourly):,} rows)")


# Leads: EventBookingLeadCreated (see notebook 11.16.4 / 11.13). Platform and
# event-size field names aren't guaranteed, so sample a few real documents
# and detect them the same way build_takeout() detects the money field.
LEAD_PLATFORM_CANDIDATES = ["platform", "service", "source", "provider", "venue"]
LEAD_SIZE_SUBSTRINGS = ["partysize", "guests", "headcount", "attendee", "covers", "size"]


def _detect_lead_fields():
    sample = list(db["events"].find({"type": "EventBookingLeadCreated"}).limit(10))
    if not sample:
        return None, None
    cols = set()
    for doc in sample:
        cols.update(doc.keys())
    platform_field = next((c for c in LEAD_PLATFORM_CANDIDATES if c in cols), None)
    size_field = next((c for c in sorted(cols) if any(k in c.lower() for k in LEAD_SIZE_SUBSTRINGS)), None)
    return platform_field, size_field


def build_leads(excluded_ids):
    """Leads captured -> leads_monthly.parquet + leads_platform.parquet"""
    platform_field, size_field = _detect_lead_fields()
    print(f"  lead platform field: {platform_field!r}, event-size field: {size_field!r}")

    fields = {"_id": 0, "businessId": 1, "createdAt": 1}
    if platform_field:
        fields[platform_field] = 1
    if size_field:
        fields[size_field] = 1

    match = {"type": "EventBookingLeadCreated"}
    if config.DEV_LIMIT:
        cursor = db["events"].aggregate(
            [{"$match": match}, {"$sample": {"size": config.DEV_LIMIT}}, {"$project": fields}],
            allowDiskUse=True,
        )
    else:
        cursor = db["events"].find(match, fields)
    leads = pd.DataFrame(list(cursor))
    print(f"  pulled {len(leads):,} leads")

    leads = drop_epoch_dates(leads, "createdAt", "leads")
    if "businessId" in leads.columns:
        leads = leads[~leads["businessId"].astype(str).isin(excluded_ids)]
    if platform_field and platform_field in leads.columns:
        leads = leads.rename(columns={platform_field: "platform"})
    if size_field and size_field in leads.columns:
        leads = leads.rename(columns={size_field: "event_size"})

    restaurants = load("restaurants", fields={"_id": 1, "name": 1})

    monthly = m_leads.build_monthly(leads, restaurants)
    out = config.Data_Directory / "leads_monthly.parquet"
    monthly.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(monthly):,} restaurant-months)")

    platform = m_leads.build_platform(leads)
    out2 = config.Data_Directory / "leads_platform.parquet"
    platform.to_parquet(out2, index=False)
    print(f"  wrote {out2}  ({len(platform):,} rows)")


# Conversation tags: tag ObjectIds in each conversation's `tags` array,
# mapped to names via the `tags` collection (see notebook 11.16.2). Channel
# (call vs. text) is inferred from conversationSummary, same as the notebook.
tags_convo_fields = {
    "_id": 0,
    "restaurantId": 1,
    "dateCreated": 1,
    "tags": 1,
    "conversationSummary": 1,
}


def build_tags(excluded_ids):
    """Conversation tags -> tags_monthly.parquet + tags_by_channel.parquet"""
    if config.DEV_LIMIT:
        cursor = db["conversations"].aggregate(
            [{"$sample": {"size": config.DEV_LIMIT}}, {"$project": tags_convo_fields}],
            allowDiskUse=True,
        )
    else:
        cursor = db["conversations"].find({}, tags_convo_fields)
    convo = pd.DataFrame(list(cursor))
    print(f"  pulled {len(convo):,} conversations for tags")

    convo = drop_epoch_dates(convo, "dateCreated", "tags conversations")
    convo = convo[~convo["restaurantId"].astype(str).isin(excluded_ids)]

    tags_lookup_df = load("tags", fields={"_id": 1, "name": 1})
    tag_lookup = dict(zip(tags_lookup_df["_id"], tags_lookup_df["name"]))
    convo["tag_names"] = convo["tags"].apply(
        lambda tags: [tag_lookup.get(t, str(t)) for t in tags] if isinstance(tags, list) else []
    )

    restaurants = load("restaurants", fields={"_id": 1, "name": 1})

    monthly = m_tags.build_monthly(convo, restaurants)
    out = config.Data_Directory / "tags_monthly.parquet"
    monthly.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(monthly):,} restaurant-months)")

    tag_counts = m_tags.build_tag_counts(convo)
    out2 = config.Data_Directory / "tags_by_channel.parquet"
    tag_counts.to_parquet(out2, index=False)
    print(f"  wrote {out2}  ({len(tag_counts):,} rows)")


# Transfers: notebook 11.7. Attempt = TransferStart/TransferAttempt/
# CallTransfer, or CallEnd with reason=="TRANSFER". Answered = same
# conversationId also has a TransferComplete event. These events carry a
# conversationId but no restaurantId/createdAt, so we join to a lookup
# pulled from the conversations collection.
TRANSFER_TYPES = ["TransferStart", "TransferAttempt", "CallTransfer", "TransferComplete", "CallEnd"]
transfer_event_fields = {"_id": 0, "type": 1, "reason": 1, "conversationId": 1}
transfer_conv_lookup_fields = {"_id": 1, "restaurantId": 1, "dateCreated": 1}


def build_transfers(excluded_ids):
    """Unanswered-transfer rate -> transfers_monthly.parquet"""
    match = {"type": {"$in": TRANSFER_TYPES}}
    if config.DEV_LIMIT:
        cursor = db["events"].aggregate(
            [
                {"$match": match},
                {"$sample": {"size": config.DEV_LIMIT}},
                {"$project": transfer_event_fields},
            ],
            allowDiskUse=True,
        )
    else:
        cursor = db["events"].find(match, transfer_event_fields)
    events = pd.DataFrame(list(cursor))
    print(f"  pulled {len(events):,} transfer-related events")
    if events.empty:
        print("  no transfer events found -- skipping transfers snapshot")
        return

    # Lookup: conversationId ("_id" on conversations) -> restaurantId/dateCreated.
    # Sampled the same way as everything else in dev mode -- some transfer
    # events may not find a match if their conversation wasn't in the sample.
    if config.DEV_LIMIT:
        lookup_cursor = db["conversations"].aggregate(
            [{"$sample": {"size": config.DEV_LIMIT}}, {"$project": transfer_conv_lookup_fields}],
            allowDiskUse=True,
        )
    else:
        lookup_cursor = db["conversations"].find({}, transfer_conv_lookup_fields)
    conv_lookup = pd.DataFrame(list(lookup_cursor))
    conv_lookup = conv_lookup.rename(columns={"_id": "conversationId"})
    conv_lookup["conversationId"] = conv_lookup["conversationId"].astype(str)
    events["conversationId"] = events["conversationId"].astype(str)
    print(f"  pulled {len(conv_lookup):,} conversations for the transfer lookup")

    conv_lookup = drop_epoch_dates(conv_lookup, "dateCreated", "transfer lookup conversations")

    # No demo/test filtering existed here before -- drop excluded restaurants'
    # conversations from the lookup so their transfer events can't join and
    # get dropped by build_monthly's dropna(subset=["restaurantId"]).
    conv_lookup = conv_lookup[~conv_lookup["restaurantId"].astype(str).isin(excluded_ids)]

    restaurants = load("restaurants", fields={"_id": 1, "name": 1})

    monthly = m_transfers.build_monthly(events, conv_lookup, restaurants)
    out = config.Data_Directory / "transfers_monthly.parquet"
    monthly.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(monthly):,} restaurant-months)")


def main():
    print("Connecting to MongoDB…")
    print("Loading restaurant exclusion list (demo_restaurants + demo/test-named restaurants)…")
    excluded_ids = load_excluded_restaurant_ids()
    print("Building conversations snapshot…")
    build_conversations(excluded_ids)
    print("Building reservations snapshot…")
    build_reservations(excluded_ids)
    print("Building takeout snapshot…")
    build_takeout(excluded_ids)
    print("Building leads snapshot…")
    build_leads(excluded_ids)
    print("Building tags snapshot…")
    build_tags(excluded_ids)
    print("Building transfers snapshot…")
    build_transfers(excluded_ids)
    print("Done.")


if __name__ == "__main__":
    main()