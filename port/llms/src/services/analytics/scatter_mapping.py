from __future__ import annotations

from typing import Iterable


def map_scores_to_scatter_points(
    rows: Iterable[dict],
    *,
    x_key: str,
    y_key: str,
    label_key: str = "label",
) -> list[dict]:
    """
    Build normalized scatter points from analytics rows.
    Returns plain dicts so mapping can be unit-tested independently from UI controls.
    """
    points: list[dict] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        try:
            points.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "label": str(row.get(label_key, "")),
                }
            )
        except (TypeError, ValueError):
            continue
    return points
