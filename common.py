"""Small helpers shared by every script: paths, device, seeding, and a persistent key-value cache."""
import json
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "figures"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class JsonCache:
    """SQLite table key -> JSON dict. Survives restarts, so no genome is ever evaluated twice."""

    def __init__(self, table, path=None):
        path = Path(path or RUNS_DIR / "cache.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table = "".join(ch if ch.isalnum() else "_" for ch in table)
        self.db = sqlite3.connect(str(path))
        self.db.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (key TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

    def get(self, key):
        row = self.db.execute(f"SELECT value FROM {self.table} WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.db.execute(f"INSERT OR REPLACE INTO {self.table} VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()
