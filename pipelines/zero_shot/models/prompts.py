"""Shared facade analysis prompts and response parsing for all VLM models.

SIMPLE_PROMPT: Legacy. Outputs one dominant material + WWR.
DETAILED_PROMPT: GT-aligned. Outputs primary + optional secondary material with
    proportion, plus back wall material and WWR. Matches the curated dataset
    columns: Front_wall, Sec_front_wall, Proportion, Back_wall, WWR.
"""

import json
import re

MATERIAL_LIST = """\
   - Brick
   - Stucco
   - Vinyl
   - Decorative stone
   - Aluminum Composite
   - Metal
   - Fibercement"""

# Current prompt (single dominant material)
SIMPLE_PROMPT = f"""Analyze this building facade image and provide:

1. PRIMARY FACADE MATERIAL: Identify the dominant wall cladding material visible on the front facade. Choose exactly one from this list:
{MATERIAL_LIST}

2. WINDOW-TO-WALL RATIO (WWR): Estimate the ratio of total window/glazing area to total wall area on the visible facade. Express as a decimal between 0.0 and 1.0.

Respond in exactly this JSON format and nothing else:
{{"material": "<material>", "wwr": <number>}}"""

# GT-aligned prompt (primary + secondary material with proportion)
DETAILED_PROMPT = f"""Analyze this building facade image. Identify ALL visible wall cladding materials on the front facade and estimate the window-to-wall ratio.

MATERIAL OPTIONS (choose only from this list):
{MATERIAL_LIST}

INSTRUCTIONS:
1. FRONT WALL (primary): The dominant cladding material covering the largest area of the front facade. Choose exactly one from the list above.
2. SECONDARY FRONT WALL: If a second distinct cladding material is clearly visible on the front facade (e.g. a different material on the ground floor vs upper floors, or a side section), identify it. Use null if only one material is present.
3. PROPORTION: The approximate area fraction of the PRIMARY material relative to the total front facade cladding area. Use one of these values: "1" (100% primary, no secondary), "3/4", "2/3", "1/2", "1/3". If secondary is null, proportion must be "1".
4. BACK WALL: If the rear or side wall is partially visible and uses a different material, identify it. Use null if not visible.
5. WWR: Estimate the window-to-wall ratio (total glazing area / total wall area) on the visible front facade as a decimal between 0.0 and 1.0.

Respond in exactly this JSON format and nothing else:
{{"front_wall": "<material>", "sec_front_wall": <"material" or null>, "proportion": "<fraction>", "back_wall": <"material" or null>, "wwr": <number>}}"""


def _coerce_str(value) -> str:
    """Return stripped string, or empty string for None / null / empty."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("null", "none", ""):
        return ""
    return s


def _coerce_wwr(value) -> float:
    try:
        wwr = float(value)
    except (TypeError, ValueError):
        return -1.0
    if 0.0 <= wwr <= 1.0:
        return wwr
    if 1.0 < wwr <= 100.0:
        return wwr / 100.0
    return -1.0


def parse_detailed_response(raw: str) -> dict:
    """Parse DETAILED_PROMPT JSON response into a normalized dict.

    Returned keys: front_wall, sec_front_wall, proportion, back_wall, wwr.
    Robust to missing fields, null values, trailing text around the JSON,
    and minor numeric formatting quirks."""
    result = {
        "front_wall": "Unknown",
        "sec_front_wall": "",
        "proportion": "1",
        "back_wall": "",
        "wwr": -1.0,
    }

    parsed = None
    # First try the whole string, then fall back to the first {...} block.
    for candidate in (raw, None):
        text = candidate if candidate is not None else None
        if text is None:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                break
            text = match.group()
        try:
            parsed = json.loads(text)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    if isinstance(parsed, dict):
        front = _coerce_str(parsed.get("front_wall")) or _coerce_str(parsed.get("material"))
        result["front_wall"] = front or "Unknown"
        result["sec_front_wall"] = _coerce_str(parsed.get("sec_front_wall"))
        prop = _coerce_str(parsed.get("proportion")) or "1"
        result["proportion"] = prop
        result["back_wall"] = _coerce_str(parsed.get("back_wall"))
        result["wwr"] = _coerce_wwr(parsed.get("wwr"))
        return result

    # Regex fallback when JSON parsing fails outright.
    def _pick(key: str) -> str:
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw)
        return m.group(1).strip() if m else ""

    front = _pick("front_wall") or _pick("material")
    if front:
        result["front_wall"] = front
    sec = _pick("sec_front_wall")
    if sec and sec.lower() not in ("null", "none"):
        result["sec_front_wall"] = sec
    prop = _pick("proportion")
    if prop:
        result["proportion"] = prop
    back = _pick("back_wall")
    if back and back.lower() not in ("null", "none"):
        result["back_wall"] = back
    wwr_match = re.search(r'"wwr"\s*:\s*([\d.]+)', raw)
    if wwr_match:
        result["wwr"] = _coerce_wwr(wwr_match.group(1))
    return result
