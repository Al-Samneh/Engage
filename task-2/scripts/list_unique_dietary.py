"""
Utility script to inspect the unique values stored in user_data.csv -> dietary_restrictions.
Run with `python task-2/scripts/list_unique_dietary.py` to print both raw and normalized sets.
"""
from pathlib import Path

import pandas as pd

from config import DATA_DIR


def main():
    csv_path = Path(DATA_DIR) / "user_data.csv"
    if not csv_path.exists():
        raise SystemExit(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    col = "dietary_restrictions"
    raw_values = (
        df[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
    normalized = raw_values.str.lower()

    print(f"Raw unique values ({len(raw_values.unique())}):")
    for value in sorted(raw_values.unique()):
        print(f"  - {value}")

    print(f"\nNormalized unique values ({len(normalized.unique())}):")
    for value in sorted(normalized.unique()):
        print(f"  - {value}")


if __name__ == "__main__":
    main()

