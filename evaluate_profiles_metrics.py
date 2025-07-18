"""
evaluate_profiles_metrics.py

Given:
  - FINAL.csv (ground-truth columns: Name, Type, Country, YearFounded, Website,
               ShortDescription, SourceText)
  - profiles.json (model output, each object with keys: name, country, year_founded,
                   website, type, short_description, full_profile, etc.)

This script will:
  1. Load FINAL.csv as a DataFrame (strings for all columns), forcing latin-1 encoding.
  2. Load profiles.json as a DataFrame.
  3. Merge them on institution Name / name.
  4. Compute accuracy, precision, recall, F₁, E‐measure (E = 1 − F₁) for each field:
     - Type
     - Country
     - YearFounded
     - Website
     - ShortDescription
     - SourceText
  5. Save all metrics into metrics_summary.csv and plot a grouped bar chart.

Requirements:
    pip install pandas scikit-learn matplotlib
"""

import os
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# —— CONFIGURATION —— #

# 1) Path to FINAL.csv (ground-truth). Adjust if needed.
TRUE_CSV_PATH = "FINAL.csv"

# 2) Path to the JSON file produced by profiles.py. Adjust if needed.
JSON_PATH = "profiles.json"

# 3) CSV column names (ground truth)
TRUE_NAME_COL            = "Name"
TRUE_TYPE_COL            = "Type"
TRUE_COUNTRY_COL         = "Country"
TRUE_YEAR_FOUNDED_COL    = "YearFounded"
TRUE_WEBSITE_COL         = "Website"
TRUE_SHORT_DESC_COL      = "ShortDescription"
TRUE_FULL_PROFILE_COL    = "SourceText"

# 4) JSON key names (predictions)
PRED_NAME_KEY            = "name"
PRED_TYPE_KEY            = "type"
PRED_COUNTRY_KEY         = "country"
PRED_YEAR_FOUNDED_KEY    = "year_founded"
PRED_WEBSITE_KEY         = "website"
PRED_SHORT_DESC_KEY      = "short_description"
PRED_FULL_PROFILE_KEY    = "full_profile"


def load_true_csv(csv_path: str) -> pd.DataFrame:
    """
    Load FINAL.csv, cast everything to string, normalize "null"/"None" to pd.NA.
    Uses latin-1 encoding (instead of UTF-8) to avoid byte-decoding errors.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1")
    except FileNotFoundError:
        print(f"❌ ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR reading CSV '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = [
        TRUE_NAME_COL,
        TRUE_TYPE_COL,
        TRUE_COUNTRY_COL,
        TRUE_YEAR_FOUNDED_COL,
        TRUE_WEBSITE_COL,
        TRUE_SHORT_DESC_COL,
        TRUE_FULL_PROFILE_COL,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ ERROR: CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Normalize string versions of null to pandas' NA
    df = df.replace({"null": pd.NA, "None": pd.NA, "NULL": pd.NA})
    return df


def load_pred_json(json_path: str) -> pd.DataFrame:
    """
    Load profiles.json (a list of objects). Return DataFrame.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR parsing JSON '{json_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"❌ ERROR: Expected JSON array, got {type(data)}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(data)
    return df


def prepare_and_merge(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename/align columns so we can merge on 'name', then do an inner join.
    Returns merged DataFrame with columns:
      name, type_true, type_pred, country_true, country_pred, year_founded_true,
      year_founded_pred, website_true, website_pred, short_desc_true, short_desc_pred,
      full_profile_true, full_profile_pred.
    """
    # 1) Rename ground-truth columns
    true_df = true_df.rename(
        columns={
            TRUE_NAME_COL: "name",
            TRUE_TYPE_COL: "type_true",
            TRUE_COUNTRY_COL: "country_true",
            TRUE_YEAR_FOUNDED_COL: "year_founded_true",
            TRUE_WEBSITE_COL: "website_true",
            TRUE_SHORT_DESC_COL: "short_desc_true",
            TRUE_FULL_PROFILE_COL: "full_profile_true",
        }
    )

    # 2) Rename prediction columns
    pred_df = pred_df.rename(
        columns={
            PRED_NAME_KEY: "name",
            PRED_TYPE_KEY: "type_pred",
            PRED_COUNTRY_KEY: "country_pred",
            PRED_YEAR_FOUNDED_KEY: "year_founded_pred",
            PRED_WEBSITE_KEY: "website_pred",
            PRED_SHORT_DESC_KEY: "short_desc_pred",
            PRED_FULL_PROFILE_KEY: "full_profile_pred",
        }
    )

    # 3) Cast year_founded to numeric Int64 if possible
    true_df["year_founded_true"] = pd.to_numeric(true_df["year_founded_true"], errors="coerce").astype("Int64")
    pred_df["year_founded_pred"] = pd.to_numeric(pred_df["year_founded_pred"], errors="coerce").astype("Int64")

    # 4) Merge on 'name' (inner join)
    merged = pd.merge(true_df, pred_df, on="name", how="inner")
    if merged.shape[0] == 0:
        print(
            "⚠️ WARNING: Merge returned 0 rows. "
            "Check that 'Name' in CSV exactly matches 'name' in JSON (case/whitespace).",
            file=sys.stderr,
        )
    return merged


def compute_metrics_for_field(y_true, y_pred, average="macro"):
    """
    Compute accuracy, precision, recall, F1, E (1-F1) for two pandas Series or lists.
    Drops pairs where either is NA or empty string.
    Treats them as strings (exact match). Returns a dict.
    """
    y_true = pd.Series(y_true).astype("string")
    y_pred = pd.Series(y_pred).astype("string")

    mask = (~y_true.isna()) & (~y_pred.isna()) & (y_true != "") & (y_pred != "")
    y_t = y_true[mask]
    y_p = y_pred[mask]

    n = int(y_t.shape[0])
    if n == 0:
        return {
            "num_evaluated": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "e_measure": None,
        }

    unique_labels = set(y_t.unique()) | set(y_p.unique())
    is_binary = len(unique_labels) == 2
    avg_type = "binary" if is_binary else average

    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, average=avg_type, zero_division=0)
    rec = recall_score(y_t, y_p, average=avg_type, zero_division=0)
    f1 = f1_score(y_t, y_p, average=avg_type, zero_division=0)
    e = 1.0 - f1

    return {
        "num_evaluated": n,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "e_measure": float(e),
    }


def main():
    # Check that both files exist
    if not os.path.exists(TRUE_CSV_PATH):
        print(f"❌ ERROR: Ground-truth CSV not found at {TRUE_CSV_PATH}.")
        sys.exit(1)
    if not os.path.exists(JSON_PATH):
        print(f"❌ ERROR: Prediction JSON not found at {JSON_PATH}.")
        sys.exit(1)

    print("🔍 Loading ground-truth CSV (FINAL.csv) with latin-1 encoding…")
    df_true = load_true_csv(TRUE_CSV_PATH)

    print("🔍 Loading prediction JSON (profiles.json)…")
    df_pred = load_pred_json(JSON_PATH)

    print("🔍 Merging on institution name…")
    merged_df = prepare_and_merge(df_true, df_pred)
    print(f"   → After merge: {merged_df.shape[0]} rows.")

    # List of (true_col, pred_col, human-readable name)
    fields = [
        ("type_true", "type_pred", "Type"),
        ("country_true", "country_pred", "Country"),
        ("year_founded_true", "year_founded_pred", "YearFounded"),
        ("website_true", "website_pred", "Website"),
        ("short_desc_true", "short_desc_pred", "ShortDescription"),
        ("full_profile_true", "full_profile_pred", "SourceText"),
    ]

    metrics_list = []
    for true_col, pred_col, label in fields:
        print(f"\n📊 Computing metrics for '{label}'…")
        m = compute_metrics_for_field(merged_df[true_col], merged_df[pred_col], average="macro")
        if m["num_evaluated"] == 0:
            print(f"   ⚠️  No non-null comparisons for '{label}'. Skipping.")
            metrics_list.append({
                "Field": label,
                "Num_Evaluated": 0,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1_Score": None,
                "E_Measure": None
            })
            continue

        print(f"   Num evaluated : {m['num_evaluated']}")
        print(f"   Accuracy      : {m['accuracy']:.4f}")
        print(f"   Precision     : {m['precision']:.4f}")
        print(f"   Recall        : {m['recall']:.4f}")
        print(f"   F1 Score      : {m['f1_score']:.4f}")
        print(f"   E‐measure (1−F1): {m['e_measure']:.4f}")

        metrics_list.append({
            "Field": label,
            "Num_Evaluated": m["num_evaluated"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1_Score": m["f1_score"],
            "E_Measure": m["e_measure"]
        })

    # Create DataFrame of metrics and save to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("metrics_summary.csv", index=False)
    print("\n✅ Metrics saved to metrics_summary.csv")

    # Plot metrics: grouped bar chart
    fields_labels = metrics_df["Field"]
    x = range(len(fields_labels))
    width = 0.15

    plt.figure(figsize=(12, 6))
    plt.bar([i - 2*width for i in x], metrics_df["Accuracy"], width, label="Accuracy")
    plt.bar([i - width for i in x], metrics_df["Precision"], width, label="Precision")
    plt.bar(x, metrics_df["Recall"], width, label="Recall")
    plt.bar([i + width for i in x], metrics_df["F1_Score"], width, label="F1 Score")
    plt.bar([i + 2*width for i in x], metrics_df["E_Measure"], width, label="E Measure")

    plt.xlabel("Field")
    plt.ylabel("Metric Value")
    plt.title("Classification Metrics by Field")
    plt.xticks(x, fields_labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
