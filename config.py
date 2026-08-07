"""Central configuration."""
from pathlib import Path

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "test"

Base_Directory = Path(__file__).resolve().parent
Data_Directory = Base_Directory / "data"
Data_Directory.mkdir(exist_ok=True)

DEV_LIMIT = 200000
MIN_CONVERSATIONS = 30
PALETTE = {"primary": "#2E5BFF", "accent": "#00C2A8", "warn": "#F5A623", "muted": "#8A94A6"}
