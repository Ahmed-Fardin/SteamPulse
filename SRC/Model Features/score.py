"""
SteamPulse — Catalog Scoring Pipeline
======================================

Purpose:
    Use the verified release-candidate model to score all games
    available in the engineered feature dataset.

Input:
    Data/Processed/steam_features_pricefixed.csv
    Data/Processed/steam_modeling_final_prices_fixed.csv
    Models/steampulse_release_candidate.joblib

Output:
    Data/Processed/steampulse_catalog_predictions.csv
    reports/steampulse_catalog_scoring_summary.json

Important:
    - Does not train a model.
    - Does not use labels to generate predictions.
    - Does not evaluate the test partition.
    - Preserves observed outcomes separately from predictions.
    - Uses the saved decision threshold.
    - Processes data in batches.
    - Does not overwrite an existing prediction file.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT
    / "Data"
    / "Processed"
    / "steam_features_pricefixed.csv"
)

TEXT_PATH = (
    ROOT
    / "Data"
    / "Processed"
    / "steam_modeling_final_prices_fixed.csv"
)

MODEL_PATH = (
    ROOT
    / "Models"
    / "steampulse_release_candidate.joblib"
)

OUTPUT_PATH = (
    ROOT
    / "Data"
    / "Processed"
    / "steampulse_catalog_predictions.csv"
)

REPORT_DIR = ROOT / "reports"

SUMMARY_PATH = (
    REPORT_DIR
    / "steampulse_catalog_scoring_summary.json"
)

BATCH_SIZE = 2_000

TEXT_COLUMNS = [
    "steam_appid",
    "name",
    "short_description",
    "about_the_game",
]


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def split_multilabel(value):
    """Convert pipe-separated genres or categories to a list."""

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


# ============================================================
# CHECK FILES
# ============================================================

section("STEAMPULSE CATALOG SCORING")

print(f"Project root: {ROOT}")
print(f"Model artifact: {MODEL_PATH}")
print(f"Batch size: {BATCH_SIZE:,}")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for path in [
    FEATURE_PATH,
    TEXT_PATH,
    MODEL_PATH,
]:

    require(
        path.is_file(),
        f"Required input file not found: {path}",
    )

# Never silently overwrite a previous result.
require(
    not OUTPUT_PATH.exists(),
    (
        f"Prediction output already exists:\n{OUTPUT_PATH}\n"
        "Preserve the existing file. Inspect it before deciding "
        "whether to use another output filename."
    ),
)


# ============================================================
# LOAD VERIFIED MODEL
# ============================================================

section("LOADING VERIFIED MODEL")

bundle = joblib.load(MODEL_PATH)

required_bundle_keys = [
    "project",
    "model",
    "decision_threshold",
    "score_type",
    "preprocessing",
    "feature_schema",
]

missing_keys = [
    key
    for key in required_bundle_keys
    if key not in bundle
]

require(
    not missing_keys,
    f"Model bundle is missing keys: {missing_keys}",
)

require(
    bundle["project"] == "SteamPulse",
    "Unexpected project identifier.",
)

require(
    bundle["score_type"] == "decision_function",
    "Unexpected model score type.",
)

model = bundle["model"]

preprocessing = bundle["preprocessing"]

schema = bundle["feature_schema"]

threshold = float(
    bundle["decision_threshold"]
)

print(f"Model: {bundle.get('model_name', 'Unknown')}")
print(f"C: {bundle['hyperparameters']['C']}")
print(f"Decision threshold: {threshold:.10f}")
print(f"Expected feature count: {model.coef_.shape[1]:,}")


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING CATALOG FEATURES")

features = pd.read_csv(
    FEATURE_PATH,
    low_memory=False,
)

required_feature_columns = [
    "steam_appid",
    *schema["numeric_features"],
    *schema["categorical_features"],
    *schema["multilabel_features"],
]

missing_feature_columns = sorted(
    set(required_feature_columns) - set(features.columns)
)

require(
    not missing_feature_columns,
    f"Feature dataset is missing columns: {missing_feature_columns}",
)

if "split" not in features.columns:
    features["split"] = "unassigned"

if "hit" not in features.columns:
    features["hit"] = np.nan

features["steam_appid"] = pd.to_numeric(
    features["steam_appid"],
    errors="coerce",
)

features = features.dropna(
    subset=["steam_appid"]
).copy()

features["steam_appid"] = (
    features["steam_appid"].astype("int64")
)

require(
    not features["steam_appid"].duplicated().any(),
    "Duplicate Steam app IDs exist in the feature dataset.",
)

# Retain outcomes only as separate metadata.
features["observed_hit"] = pd.to_numeric(
    features["hit"],
    errors="coerce",
)

features["observed_hit"] = (
    features["observed_hit"]
    .where(features["observed_hit"].isin([0, 1]))
)

features["has_observed_outcome"] = (
    features["observed_hit"].notna()
)

print(f"Games with engineered features: {len(features):,}")

print(
    "Games with observed outcomes: "
    f"{features['has_observed_outcome'].sum():,}"
)

print(
    "Games without observed outcomes: "
    f"{(~features['has_observed_outcome']).sum():,}"
)


# ============================================================
# LOAD GAME TEXT
# ============================================================

section("LOADING GAME TITLES AND DESCRIPTIONS")

text_data = pd.read_csv(
    TEXT_PATH,
    usecols=TEXT_COLUMNS,
    low_memory=False,
)

text_data = text_data.rename(
    columns={
        "name": "text_name",
    }
)

text_data["steam_appid"] = pd.to_numeric(
    text_data["steam_appid"],
    errors="coerce",
)

text_data = text_data.dropna(
    subset=["steam_appid"]
).copy()

text_data["steam_appid"] = (
    text_data["steam_appid"].astype("int64")
)

require(
    not text_data["steam_appid"].duplicated().any(),
    "Duplicate Steam app IDs found in the text dataset.",
)

data = features.merge(
    text_data,
    on="steam_appid",
    how="left",
    validate="one_to_one",
)

data["name"] = (
    data["name"].fillna("")
    if "name" in data.columns
    else data["text_name"].fillna("")
)

data["text_name"] = (
    data["text_name"]
    .fillna(data["name"])
    .fillna("")
)

data["short_description"] = (
    data["short_description"]
    .fillna("")
    .astype(str)
)

data["about_the_game"] = (
    data["about_the_game"]
    .fillna("")
    .astype(str)
)

print(f"Games after text join: {len(data):,}")


# ============================================================
# PREPARE OUTPUT FIELDS
# ============================================================

section("PREPARING OUTPUT COLUMNS")

# Include only fields useful for ranking, filtering and exploration.
display_columns = [
    "steam_appid",
    "name",
    "release_year",
    "release_month",
    "main_genre",
    "main_category",
    "main_developer",
    "main_publisher",
    "is_free",
    "platform_count",
    "genre_count",
    "category_count",
    "price_log1p_final",
    "split",
    "has_observed_outcome",
    "observed_hit",
]

display_columns = [
    column
    for column in display_columns
    if column in data.columns
]

output_frames = []

total_rows = len(data)

score_min = None
score_max = None

predicted_hit_count = 0

print(f"Rows to score: {total_rows:,}")


# ============================================================
# SCORE IN BATCHES
# ============================================================

section("SCORING CATALOG")

numeric_features = schema["numeric_features"]

categorical_features = schema["categorical_features"]

multilabel_features = schema["multilabel_features"]

structured_columns = (
    numeric_features + categorical_features
)

total_batches = (
    total_rows + BATCH_SIZE - 1
) // BATCH_SIZE

for batch_number, start in enumerate(
    range(0, total_rows, BATCH_SIZE),
    start=1,
):

    end = min(
        start + BATCH_SIZE,
        total_rows,
    )

    batch = data.iloc[start:end].copy()

    # --------------------------------------------------------
    # Structured features
    # --------------------------------------------------------

    X_structured = preprocessing[
        "structured_preprocessor"
    ].transform(
        batch[structured_columns]
    )

    X_structured = sparse.csr_matrix(
        X_structured,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Multi-label features
    # --------------------------------------------------------

    multilabel_matrices = []

    for column in multilabel_features:

        encoder = preprocessing[
            "multilabel_encoders"
        ][column]

        values = (
            batch[column]
            .apply(split_multilabel)
            .tolist()
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            matrix = encoder.transform(values)

        matrix = sparse.csr_matrix(
            matrix,
            dtype=np.float32,
        )

        multilabel_matrices.append(matrix)

    # --------------------------------------------------------
    # Description and title text
    # --------------------------------------------------------

    descriptions = (
        batch["short_description"]
        + " "
        + batch["about_the_game"]
    ).tolist()

    titles = (
        batch["text_name"]
        .fillna("")
        .astype(str)
    ).tolist()

    X_description = preprocessing[
        "description_vectorizer"
    ].transform(descriptions)

    X_title = preprocessing[
        "title_vectorizer"
    ].transform(titles)

    # --------------------------------------------------------
    # Combine in exactly the same feature order as training
    # --------------------------------------------------------

    X_batch = sparse.hstack(
        [
            X_structured,
            *multilabel_matrices,
            X_description,
            X_title,
        ],
        format="csr",
        dtype=np.float32,
    )

    require(
        X_batch.shape[1] == model.coef_.shape[1],
        (
            f"Batch {batch_number} feature mismatch: "
            f"{X_batch.shape[1]} versus {model.coef_.shape[1]}."
        ),
    )

    # Decision scores must be compared with the saved threshold.
    scores = model.decision_function(
        X_batch
    )

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    batch_output = batch[display_columns].copy()

    batch_output["decision_score"] = scores.astype(float)

    batch_output["predicted_hit"] = predictions

    # Keep the observed outcome separate from the prediction.
    # The model has NOT been evaluated on this scoring batch.
    batch_output["prediction_label"] = np.where(
        predictions == 1,
        "HIT",
        "NOT FLAGGED AS HIT",
    )

    output_frames.append(
        batch_output
    )

    predicted_hit_count += int(
        predictions.sum()
    )

    batch_min = float(np.min(scores))
    batch_max = float(np.max(scores))

    score_min = (
        batch_min
        if score_min is None
        else min(score_min, batch_min)
    )

    score_max = (
        batch_max
        if score_max is None
        else max(score_max, batch_max)
    )

    print(
        f"Batch {batch_number}/{total_batches} "
        f"| rows {start + 1:,}-{end:,} "
        f"| flagged HIT: {int(predictions.sum()):,}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

section("SAVING PREDICTIONS")

results = pd.concat(
    output_frames,
    ignore_index=True,
)

require(
    len(results) == total_rows,
    (
        f"Output row mismatch: expected {total_rows}, "
        f"created {len(results)}."
    ),
)

require(
    not results["steam_appid"].duplicated().any(),
    "Duplicate IDs found in scored output.",
)

results = results.sort_values(
    "decision_score",
    ascending=False,
).reset_index(drop=True)

results.insert(
    0,
    "model_rank",
    np.arange(1, len(results) + 1),
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

summary = {
    "project": "SteamPulse",
    "model_artifact": str(MODEL_PATH),
    "scored_dataset": str(OUTPUT_PATH),
    "rows_scored": int(len(results)),
    "unique_games": int(results["steam_appid"].nunique()),
    "batch_size": int(BATCH_SIZE),
    "decision_threshold": float(threshold),
    "decision_score_min": float(score_min),
    "decision_score_max": float(score_max),
    "predicted_hit_count": int(predicted_hit_count),
    "predicted_hit_fraction": float(
        predicted_hit_count / len(results)
    ) if len(results) else 0.0,
    "observed_outcome_count": int(
        results["has_observed_outcome"].sum()
    ),
    "unobserved_outcome_count": int(
        (~results["has_observed_outcome"]).sum()
    ),
    "test_set_used_for_evaluation": False,
    "score_interpretation": (
        "decision_function score; not a calibrated probability"
    ),
    "label_interpretation": (
        "HIT denotes the model's classification under the saved "
        "threshold. It is not a statement of commercial success."
    ),
}

with SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

section("CATALOG SCORING COMPLETE")

print(f"Rows scored: {len(results):,}")
print(f"Unique games: {results['steam_appid'].nunique():,}")
print(
    "Games with observed outcomes: "
    f"{summary['observed_outcome_count']:,}"
)

print(
    "Games without observed outcomes: "
    f"{summary['unobserved_outcome_count']:,}"
)

print(
    f"Predicted HIT: {predicted_hit_count:,} "
    f"({summary['predicted_hit_fraction']:.2%})"
)

print(f"Minimum decision score: {score_min:.6f}")
print(f"Maximum decision score: {score_max:.6f}")

print(f"\nPrediction CSV: {OUTPUT_PATH}")
print(f"Summary report: {SUMMARY_PATH}")

print("\nNo model training was performed.")
print("No test-set evaluation was performed.")
print("Catalog scoring completed successfully.")