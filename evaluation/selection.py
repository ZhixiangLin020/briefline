"""Pure validation-only checkpoint selection helper."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional


def select_best_finetuned_by_validation(
    rows: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select only by validation combo; test metrics never affect ranking."""

    candidates = []
    for row in rows:
        if row.get("model_alias") == "base":
            continue
        value = row.get("validation_combo_mover_score")
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            candidates.append((value, dict(row)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]

