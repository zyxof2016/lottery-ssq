from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"历史数据文件不存在: {source}")
    return pd.read_csv(source, dtype={"period": "string", "date": "string"})

