"""Conversation tag metrics (spec: Conversation Tags -- sections 10/11.11/11.16.2).

Tags are ObjectIds in each conversation's `tags` array, mapped to names via
the `tags` collection joined on `_id` (raw ObjectId keys, no string
conversion -- same as the notebook). Channel (call vs. text) isn't a stored
field -- `direction` is inbound/outbound, not call/text -- so text is
detected the same way as the notebook's TA_4 reference: a
conversationSummary flagged "conversation happened over text".

Same additive-snapshot idea as the other sections: build_monthly stores
counts per (restaurant, month) for conversations, channel, and each of the
four tag rates the spec asks for (speak-to-human, miscommunication,
frustrated, cancellation), so rates are always summed-then-divided in the
app rather than averaged ahead of time in data_prep.py.
"""
import numpy as np
import pandas as pd

from Metrics.common import percentile_groups

# Exact tag-name sets (not loose regex) -- matches the real tag vocabulary,
# avoiding false positives like "notify-host" matching a loose "host" regex.
TAG_FLAG_SETS = {
    "speak_human": {"speak to human"},
    "miscommunication": {"miscommunication", "wrong info"},
    "frustrated": {"frustrated", "angry"},
    "cancellation": {"cancellation"},
}


def _channel(convo: pd.DataFrame) -> pd.Series:
    is_text = convo["conversationSummary"].fillna("").str.contains(
        "conversation happened over text", case=False, na=False
    )
    return np.where(is_text, "Text", "Call")


def build_monthly(convo: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """One row per (restaurant, month): conversations, channel counts, tag-rate counts.

    `convo` is expected to already have a `tag_names` column (list of tag name
    strings per conversation) -- built by data_prep.py from the tags collection.
    """
    convo = convo.copy()
    convo["restaurantId"] = convo["restaurantId"].astype(str)
    convo["dateCreated"] = pd.to_datetime(convo["dateCreated"], errors="coerce", utc=True)
    convo["year_month"] = convo["dateCreated"].dt.tz_convert(None).dt.to_period("M").astype(str)

    convo["channel"] = _channel(convo)
    convo["is_text"] = convo["channel"] == "Text"
    convo["is_call"] = convo["channel"] == "Call"

    for flag_col, tagset in TAG_FLAG_SETS.items():
        lowered = {t.lower() for t in tagset}
        convo[f"tag_{flag_col}"] = convo["tag_names"].apply(
            lambda names: any(str(n).lower() in lowered for n in names)
            if isinstance(names, list) else False
        )

    monthly = convo.groupby(["restaurantId", "year_month"]).agg(
        conversations=("restaurantId", "size"),
        text_n=("is_text", "sum"),
        call_n=("is_call", "sum"),
        speak_human_n=("tag_speak_human", "sum"),
        miscommunication_n=("tag_miscommunication", "sum"),
        frustrated_n=("tag_frustrated", "sum"),
        cancellation_n=("tag_cancellation", "sum"),
    ).reset_index()

    names = restaurants.rename(columns={"_id": "restaurantId"})[["restaurantId", "name"]].copy()
    names["restaurantId"] = names["restaurantId"].astype(str)
    monthly = monthly.merge(names, on="restaurantId", how="left")
    return monthly


def build_tag_counts(convo: pd.DataFrame) -> pd.DataFrame:
    """Long-format tag counts per (month, channel, tag) -- platform-wide, for the
    top-tags-by-channel table (notebook 11.16.2's first result)."""
    convo = convo.copy()
    convo["dateCreated"] = pd.to_datetime(convo["dateCreated"], errors="coerce", utc=True)
    convo["year_month"] = convo["dateCreated"].dt.tz_convert(None).dt.to_period("M").astype(str)
    convo["channel"] = _channel(convo)

    exploded = convo.explode("tag_names").dropna(subset=["tag_names"])
    exploded = exploded[exploded["tag_names"].astype(str).str.len() > 0]

    return (
        exploded.groupby(["year_month", "channel", "tag_names"])
        .size()
        .reset_index(name="count")
        .rename(columns={"tag_names": "tag"})
    )


# --- App-side slicing helpers ----------------------------------------------

def platform_kpis(view: pd.DataFrame) -> dict:
    conversations = view["conversations"].sum()
    return {
        "conversations": int(conversations),
        "text_rate": view["text_n"].sum() / conversations if conversations else float("nan"),
        "speak_human_rate": view["speak_human_n"].sum() / conversations if conversations else float("nan"),
        "miscommunication_rate": view["miscommunication_n"].sum() / conversations if conversations else float("nan"),
        "frustrated_rate": view["frustrated_n"].sum() / conversations if conversations else float("nan"),
        "cancellation_rate": view["cancellation_n"].sum() / conversations if conversations else float("nan"),
    }


def monthly_tag_rates(view: pd.DataFrame) -> pd.DataFrame:
    """Tag rates per month, for trend lines (11.16.2's miscommunication-trend enrichment)."""
    g = view.groupby("year_month", as_index=False).agg(
        conversations=("conversations", "sum"),
        speak_human_n=("speak_human_n", "sum"),
        miscommunication_n=("miscommunication_n", "sum"),
        frustrated_n=("frustrated_n", "sum"),
        cancellation_n=("cancellation_n", "sum"),
    ).sort_values("year_month")
    for col in ["speak_human", "miscommunication", "frustrated", "cancellation"]:
        g[f"{col}_rate"] = g[f"{col}_n"] / g["conversations"].replace(0, np.nan)
    return g


def top_tags_by_channel(tag_counts_view: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N tags per channel, summed over the selection (11.16.2's channel split)."""
    g = (
        tag_counts_view.groupby(["channel", "tag"], as_index=False)["count"]
        .sum()
        .sort_values(["channel", "count"], ascending=[True, False])
    )
    return g.groupby("channel", group_keys=False).head(n).reset_index(drop=True)


def top_tags_overall(tag_counts_view: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N tags platform-wide, ignoring channel (spec: "Top tags by volume,
    ranked list, platform-wide")."""
    return (
        tag_counts_view.groupby("tag", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def _restaurant_rollup(view: pd.DataFrame) -> pd.DataFrame:
    g = view.groupby(["restaurantId", "name"], as_index=False).agg(
        conversations=("conversations", "sum"),
        speak_human_n=("speak_human_n", "sum"),
        miscommunication_n=("miscommunication_n", "sum"),
        frustrated_n=("frustrated_n", "sum"),
        cancellation_n=("cancellation_n", "sum"),
    )
    g["speak_human_rate"] = g["speak_human_n"] / g["conversations"].replace(0, np.nan)
    g["miscommunication_rate"] = g["miscommunication_n"] / g["conversations"].replace(0, np.nan)
    g["frustrated_rate"] = g["frustrated_n"] / g["conversations"].replace(0, np.nan)
    g["cancellation_rate"] = g["cancellation_n"] / g["conversations"].replace(0, np.nan)
    return g


def frustrated_outliers(view: pd.DataFrame, min_conversations: int = 30) -> pd.DataFrame:
    """Restaurants whose frustrated-tag rate is > mean + 2*std across eligible
    restaurants -- the CS-outlier flag from 11.16.2's enrichments."""
    g = _restaurant_rollup(view)
    eligible = g[g["conversations"] >= min_conversations]
    empty_cols = ["name", "conversations", "frustrated_rate"]
    if eligible.empty:
        return eligible.reindex(columns=empty_cols)
    threshold = eligible["frustrated_rate"].mean() + 2 * eligible["frustrated_rate"].std()
    flagged = eligible[eligible["frustrated_rate"] > threshold]
    return (
        flagged.sort_values("frustrated_rate", ascending=False)[empty_cols]
        .reset_index(drop=True)
    )


def top_restaurants(view: pd.DataFrame, metric: str = "conversations",
                    n: int = 10, min_conversations: int = 30) -> pd.DataFrame:
    """Top N restaurants by a chosen metric. Rate metrics require a min volume."""
    g = _restaurant_rollup(view)
    if metric != "conversations":
        g = g[g["conversations"] >= min_conversations]
    display_cols = ["name", "conversations", "speak_human_rate",
                     "miscommunication_rate", "frustrated_rate", "cancellation_rate"]
    return (
        g.sort_values(metric, ascending=False)[display_cols]
        .head(n)
        .reset_index(drop=True)
    )


def restaurant_percentiles(view: pd.DataFrame, metric: str = "speak_human_rate"):
    """Bottom-25% / top-90% restaurant lists for `metric` (spec's percentile
    spread for the speak-to-human rate)."""
    g = _restaurant_rollup(view)
    return percentile_groups(g, metric)
