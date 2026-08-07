"""
Central configuration. Everything that changes between machines or over time lives here so nothing else has random voodoo magic values in here. This is the only file that should be edited to change configuration values.

Or we can have the user override these values with environment variables, but that is not implemented yet. Perhaps >.<
"""

from pathlib import Path

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "test"

Base_Directory = Path(__file__).resolve().parent
Data_Directory = Base_Directory / "data"
Data_Directory.mkdir(exist_ok=True)

# Cap the pull during dev so we don't have to wait forever for the data to load. Set to None to pull all data.

DEV_LIMIT = None

# Minimum conversation length to consider for analysis. This is to filter out spammy conversations that are too short to be meaningful.
MIN_CONVERSATIONS = 30

PALETTE = {"primary": "#2E5BFF", "accent": "#00C2A8", "warn": "#F5A623", "muted": "#8A94A6"}
