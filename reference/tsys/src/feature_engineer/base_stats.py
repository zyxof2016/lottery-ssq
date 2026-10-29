from __future__ import annotations

import numpy as np
import pandas as pd


RED_COLUMNS = [f"r{i}" for i in range(1, 7)]


def occurrence_matrix(frame: pd.DataFrame, color: str = "red") -> np.ndarray:
    if color not in {"red", "blue"}:
        raise ValueError("color 必须为 red 或 blue")
    number_count = 33 if color == "red" else 16
    matrix = np.zeros((len(frame), number_count), dtype=np.int8)
    if color == "red":
        for row_index, values in enumerate(frame[RED_COLUMNS].to_numpy(dtype=int)):
            matrix[row_index, values - 1] = 1
    else:
        matrix[np.arange(len(frame)), frame["blue"].to_numpy(dtype=int) - 1] = 1
    return matrix


def combination_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    reds = frame[RED_COLUMNS].to_numpy(dtype=int)
    ordered = np.sort(reds, axis=1)
    spans = ordered[:, -1] - ordered[:, 0]
    consecutive_pairs = (np.diff(ordered, axis=1) == 1).sum(axis=1)
    return pd.DataFrame({
        "sum": reds.sum(axis=1),
        "odd_count": (reds % 2 == 1).sum(axis=1),
        "big_count": (reds > 16).sum(axis=1),
        "zone1": (reds <= 11).sum(axis=1),
        "zone2": ((reds >= 12) & (reds <= 22)).sum(axis=1),
        "zone3": (reds >= 23).sum(axis=1),
        "span": spans,
        "consecutive_pairs": consecutive_pairs,
    })


def structure_profile(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    values = combination_statistics(frame)
    return {
        column: (float(values[column].mean()), max(float(values[column].std(ddof=0)), 0.5))
        for column in values.columns
    }
