"""
How the app reads snapshots:

The first load_snapshot("x") reads the parquet and remembers the results. Every later call in the session returns it instantly (given more recent database is given). Since streamlit reruns this whole app on every widget change, the cache is what make it 
feel relatively instant. 
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