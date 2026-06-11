"""Shared helpers for converting VLM detailed material predictions to
per-label ratios and a dominant material. Used by main.py, run_new_data.py
and the CSV writers to produce YOLO-comparison outputs."""

import logging
from typing import Optional

from data_loader import normalize_material

logger = logging.getLogger(__name__)


PROPORTION_VALUE = {
    "1": 1.0,
    "3/4": 0.75,
    "2/3": 2.0 / 3.0,
    "1/2": 0.5,
    "1/3": 1.0 / 3.0,
}


def _parse_proportion(proportion: Optional[str]) -> float:
    """Return primary-material fraction for a discrete proportion string.

    Unknown / missing values fall back to 1.0 (treat as single-material)."""
    if proportion is None:
        return 1.0
    key = proportion.strip()
    if key in PROPORTION_VALUE:
        return PROPORTION_VALUE[key]
    # Some VLMs may emit "1.0" or "100%" -- handle common variants.
    cleaned = key.rstrip("%").strip()
    try:
        val = float(cleaned)
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))
    except ValueError:
        logger.debug("Unknown proportion value %r, defaulting to 1.0", proportion)
        return 1.0


def compute_ratios(
    front_wall: str,
    sec_front_wall: Optional[str],
    proportion: Optional[str],
    alias_map: dict,
    labels: list[str],
) -> dict[str, float]:
    """Return a {label: ratio} dict summing to 1.0 (or 0.0 if no valid material).

    Primary gets the proportion fraction, secondary gets the remainder. If
    secondary is absent or equals primary, primary gets 1.0. Predictions that
    do not normalize to one of `labels` are dropped and remaining ratios are
    renormalized."""
    ratios = {label: 0.0 for label in labels}

    primary_norm = normalize_material(front_wall or "", alias_map)
    sec_raw = (sec_front_wall or "").strip()
    sec_norm = normalize_material(sec_raw, alias_map) if sec_raw else ""

    primary_valid = primary_norm in ratios
    sec_valid = sec_norm in ratios and sec_norm != primary_norm

    if not sec_valid:
        if primary_valid:
            ratios[primary_norm] = 1.0
        return ratios

    p = _parse_proportion(proportion)
    p = max(0.0, min(1.0, p))

    if primary_valid:
        ratios[primary_norm] += p
    ratios[sec_norm] += 1.0 - p

    total = sum(ratios.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        for k in ratios:
            ratios[k] /= total
    return ratios


def dominant_material(ratios: dict[str, float]) -> str:
    """Return the label with the largest ratio, or empty string if all zero."""
    if not ratios:
        return ""
    label, val = max(ratios.items(), key=lambda kv: kv[1])
    return label if val > 0 else ""
