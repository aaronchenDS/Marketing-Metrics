"""How the app reads snapshots — with caching so it stays fast.

@st.cache_data means: the first time load_snapshot("x") runs, Streamlit reads
the parquet and remembers the result. Every later call in the session returns
the remembered DataFrame instantly instead of re-reading the file. Since
Streamlit re-runs this whole app top-to-bottom on every widget interaction,
this cache is what makes the app feel instant.
"""
import pandas as pd
import streamlit as st

import config


@st.cache_data
def load_snapshot(name: str) -> pd.DataFrame:
    """Read data/<name>.parquet. Returns an empty frame if it doesn't exist yet."""
    path = config.Data_Directory / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
