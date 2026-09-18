import pandas as pd
from pathlib import Path

def load_data():
    folder = Path("../data")
    files = sorted(folder.glob("*.csv"))
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)
