"""Build the per-building zero-shot-versus-supervised agreement layer on the
4,668-building raw GSV deployment set (paragraph output 4).

For each raw GSV building it records, for each VLM against the supervised YOLO
model:
  - the dominant material from each pipeline,
  - whether the two pipelines agree on the dominant material,
  - a normalized L1 distance between the two seven-class material-share vectors
    (0 = identical mixtures, 1 = fully disjoint),
  - a "comparable" flag (both pipelines returned a usable material mixture).

The normalized L1 distance is 0.5 * sum_k |zs_share_k - yolo_share_k| over the
seven canonical classes. A building is "comparable" only when YOLO status is
'success' with a known dominant material and the VLM dominant material is not
Unknown; agreement and L1 are left blank otherwise.

Inputs:
  ../3_raw_gsv_4668_deployment/raw_gsv_results_{claude,gemini,gpt}.csv
      (output 3 in this package; included)
  the supervised YOLO raw GSV predictions CSV (filename, dominant_material,
      status, pct_* columns) -- NOT included in this package; set the
      YOLO_RAW_CSV environment variable to its path to re-run.
Outputs (written next to this script):
  agreement_layer_{claude,gemini,gpt}.csv

Standard library only; no API keys or credentials are used.
"""
import csv, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
# Zero-shot deployment predictions = output (3), the sibling folder in this package.
ZS_DIR = os.path.join(HERE, "..", "3_raw_gsv_4668_deployment")
# Supervised YOLO raw GSV predictions are NOT included in this package (they belong
# to the supervised pipeline). Set YOLO_RAW_CSV to that file's path to re-run.
YOLO_RAW_CSV = os.environ.get("YOLO_RAW_CSV", os.path.join(HERE, "yolo_raw_gsv_results.csv"))

LABELS = ["Brick", "Stucco", "Vinyl", "Decorative stone",
          "Aluminum Composite", "Metal", "Fibercement"]
ZS_PCT = {l: l.replace(" ", "_") + "_Pct" for l in LABELS}
YOLO_PCT = {"Brick": "pct_brick", "Stucco": "pct_stucco", "Vinyl": "pct_vinyl",
            "Decorative stone": "pct_dec_stone", "Aluminum Composite": "pct_acm",
            "Metal": "pct_metal", "Fibercement": "pct_fibercement"}
YOLO_DOM = {"acm": "Aluminum Composite", "brick": "Brick", "dec_stone": "Decorative stone",
            "fibercement": "Fibercement", "metal": "Metal", "stucco": "Stucco", "vinyl": "Vinyl"}
MODELS = ["claude", "gemini", "gpt"]
DISP = {"claude": "Claude-Sonnet-4", "gemini": "Gemini-3-Flash", "gpt": "GPT-4o"}


def strip_fname(fn):
    s = re.sub(r"\.(jpg|jpeg|png)$", "", str(fn), flags=re.IGNORECASE)
    s = re.sub(r"_(jpg|png)\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)
    return s.strip()


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def canon_zs(v):
    v = (v or "").strip()
    return v if v in LABELS else "Unknown"


def canon_yolo(v):
    return YOLO_DOM.get((v or "").strip().lower(), "Unknown")


def load_yolo():
    path = YOLO_RAW_CSV
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[strip_fname(r["filename"])] = r
    return out


def main():
    yolo = load_yolo()
    for m in MODELS:
        zpath = os.path.join(ZS_DIR, f"raw_gsv_results_{m}.csv")
        rows_out = []
        matched = comparable = agree = 0
        with open(zpath, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = strip_fname(r["Image_ID"])
                y = yolo.get(key)
                if y is None:
                    continue
                matched += 1
                zs_dom = canon_zs(r.get("Dominant_Material"))
                y_dom = canon_yolo(y.get("dominant_material"))
                y_status = (y.get("status") or "").strip()
                is_comp = (y_status == "success" and y_dom != "Unknown" and zs_dom != "Unknown")
                dom_agree = ""
                l1 = ""
                if is_comp:
                    comparable += 1
                    same = (zs_dom == y_dom)
                    dom_agree = "yes" if same else "no"
                    if same:
                        agree += 1
                    d = sum(abs(fnum(r.get(ZS_PCT[l])) / 100.0 - fnum(y.get(YOLO_PCT[l])) / 100.0)
                            for l in LABELS)
                    l1 = round(0.5 * d, 4)
                rows_out.append({
                    "Image_ID": r.get("Image_ID", ""),
                    "Address": r.get("Address", ""),
                    "zs_dominant": zs_dom,
                    "yolo_dominant": y_dom,
                    "yolo_status": y_status,
                    "comparable": "yes" if is_comp else "no",
                    "dominant_agreement": dom_agree,
                    "l1_distance": l1,
                })
        out_path = os.path.join(HERE, f"agreement_layer_{m}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Image_ID", "Address", "zs_dominant",
                                              "yolo_dominant", "yolo_status", "comparable",
                                              "dominant_agreement", "l1_distance"])
            w.writeheader()
            w.writerows(rows_out)
        rate = 100.0 * agree / comparable if comparable else 0.0
        print(f"{DISP[m]:16s} rows={len(rows_out)} matched={matched} comparable={comparable} "
              f"agree={agree} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
