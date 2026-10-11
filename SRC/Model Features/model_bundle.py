"""
SteamPulse — Saved Model Bundle Verification
=============================================

Verifies that the saved release-candidate model:
    1. Loads successfully.
    2. Contains the required preprocessing components.
    3. Reconstructs the original validation feature matrix.
    4. Produces the recorded validation metrics.
    5. Reproduces the recorded classification threshold results.

Important:
    - Does not train a model.
    - Does not modify datasets or model artifacts.
    - Does not evaluate the test partition.
    - Correctly reads validation_pr_auc and validation_roc_auc
      from the JSON written by finalize_model.py.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT / "Data" / "Processed" / "steam_features_pricefixed.csv"
)

TEXT_PATH = (
    ROOT / "Data" / "Processed" / "steam_modeling_final_prices_fixed.csv"
)

MODEL_PATH = (
    ROOT / "Models" / "steampulse_release_candidate.joblib"
)

METRICS_PATH = (
    ROOT / "reports" / "steampulse_release_candidate_validation.json"
)

REPORT_PATH = (
    ROOT / "reports" / "steampulse_bundle_smoke_test.json"
)

NUMERIC_TOLERANCE = 1e-8

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
    """Parse pipe-separated categories or genres."""

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


def compare_float(name, actual, recorded):
    difference = abs(float(actual) - float(recorded))

    passed = difference <= NUMERIC_TOLERANCE

    print(
        f"{name}: actual={float(actual):.10f}, "
        f"recorded={float(recorded):.10f}, "
        f"difference={difference:.3e}, "
        f"PASS={passed}"
    )

    return {
        "actual": float(actual),
        "recorded": float(recorded),
        "absolute_difference": float(difference),
        "passed": bool(passed),
    }


# ============================================================
# CHECK INPUT FILES
# ============================================================

section("STEAMPULSE SAVED MODEL VERIFICATION")

print(f"Project root: {ROOT}")
print(f"Model artifact: {MODEL_PATH}")

for path in [
    FEATURE_PATH,
    TEXT_PATH,
    MODEL_PATH,
    METRICS_PATH,
]:
    require(
        path.is_file(),
        f"Required file not found: {path}",
    )


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

section("LOADING SAVED MODEL")

bundle = joblib.load(MODEL_PATH)

required_bundle_keys = [
    "project",
    "model",
    "model_name",
    "hyperparameters",
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
    f"Model bundle missing keys: {missing_keys}",
)

require(
    bundle["project"] == "SteamPulse",
    "Unexpected project identifier.",
)

require(
    bundle["model_name"] == "Logistic Regression",
    "Unexpected model type.",
)

require(
    bundle["score_type"] == "decision_function",
    "Unexpected decision score type.",
)

require(
    np.isclose(
        float(bundle["hyperparameters"]["C"]),
        0.50,
    ),
    "The saved model does not use C=0.50.",
)

model = bundle["model"]
preprocessing = bundle["preprocessing"]
schema = bundle["feature_schema"]

required_preprocessing = [
    "structured_preprocessor",
    "multilabel_encoders",
    "description_vectorizer",
    "title_vectorizer",
]

missing_preprocessing = [
    key
    for key in required_preprocessing
    if key not in preprocessing
]

require(
    not missing_preprocessing,
    f"Missing preprocessing components: {missing_preprocessing}",
)

print("Model bundle loaded successfully.")
print(f"Model type: {type(model).__name__}")
print(f"C: {bundle['hyperparameters']['C']}")
print(f"Threshold: {bundle['decision_threshold']:.10f}")
print(f"Classifier features: {model.coef_.shape[1]:,}")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

section("LOADING VALIDATION DATA")

features = pd.read_csv(
    FEATURE_PATH,
    low_memory=False,
)

required_features = [
    "steam_appid",
    "split",
    "hit",
    *schema["numeric_features"],
    *schema["categorical_features"],
    *schema["multilabel_features"],
]

missing_columns = sorted(
    set(required_features) - set(features.columns)
)

require(
    not missing_columns,
    f"Missing feature columns: {missing_columns}",
)

text_data = pd.read_csv(
    TEXT_PATH,
    usecols=TEXT_COLUMNS,
    low_memory=False,
)

text_data = text_data.rename(
    columns={"name": "text_name"}
)

features["steam_appid"] = pd.to_numeric(
    features["steam_appid"],
    errors="coerce",
)

text_data["steam_appid"] = pd.to_numeric(
    text_data["steam_appid"],
    errors="coerce",
)

features = features.dropna(
    subset=["steam_appid"]
).copy()

text_data = text_data.dropna(
    subset=["steam_appid"]
).copy()

features["steam_appid"] = (
    features["steam_appid"].astype("int64")
)

text_data["steam_appid"] = (
    text_data["steam_appid"].astype("int64")
)

require(
    not text_data["steam_appid"].duplicated().any(),
    "Duplicate app IDs found in text dataset.",
)

data = features.merge(
    text_data,
    on="steam_appid",
    how="left",
    validate="many_to_one",
)

data["split"] = (
    data["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

data["hit"] = pd.to_numeric(
    data["hit"],
    errors="coerce",
)

# Exclude missing outcomes. Do not label them as negative.
# Do not use the test partition.
validation = data[
    (data["split"] == "validation")
    & data["hit"].notna()
].copy()

validation["hit"] = validation["hit"].astype(np.int8)

y_validation = validation["hit"].to_numpy()

require(
    len(validation) == 4372,
    (
        "Unexpected validation row count. "
        f"Expected 4,372, found {len(validation):,}."
    ),
)

require(
    len(np.unique(y_validation)) == 2,
    "Validation labels must contain both classes.",
)

print(f"Validation games: {len(validation):,}")
print(f"Validation HIT prevalence: {y_validation.mean():.2%}")


# ============================================================
# RECONSTRUCT STRUCTURED FEATURES
# ============================================================

section("APPLYING SAVED STRUCTURED PREPROCESSING")

structured_columns = (
    schema["numeric_features"]
    + schema["categorical_features"]
)

X_structured = preprocessing[
    "structured_preprocessor"
].transform(
    validation[structured_columns]
)

X_structured = sparse.csr_matrix(
    X_structured,
    dtype=np.float32,
)

print("Structured matrix:", X_structured.shape)


# ============================================================
# RECONSTRUCT MULTI-LABEL FEATURES
# ============================================================

section("APPLYING SAVED MULTI-LABEL ENCODERS")

multilabel_matrices = []

for column in schema["multilabel_features"]:

    print(f"Transforming {column}...")

    encoder = preprocessing[
        "multilabel_encoders"
    ][column]

    require(
        isinstance(encoder, MultiLabelBinarizer),
        f"Unexpected encoder type for {column}.",
    )

    values = (
        validation[column]
        .apply(split_multilabel)
        .tolist()
    )

    # Match the original training behavior for unseen labels.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        matrix = encoder.transform(values)

    matrix = sparse.csr_matrix(
        matrix,
        dtype=np.float32,
    )

    multilabel_matrices.append(matrix)

    print("  Shape:", matrix.shape)


# ============================================================
# RECONSTRUCT TEXT FEATURES
# ============================================================

section("APPLYING SAVED TEXT VECTORIZERS")

descriptions = (
    validation["short_description"]
    .fillna("")
    .astype(str)
    + " "
    + validation["about_the_game"]
    .fillna("")
    .astype(str)
).tolist()

titles = (
    validation["text_name"]
    .fillna("")
    .astype(str)
).tolist()

X_description = preprocessing[
    "description_vectorizer"
].transform(descriptions)

X_title = preprocessing[
    "title_vectorizer"
].transform(titles)

print("Description matrix:", X_description.shape)
print("Title matrix:", X_title.shape)


# ============================================================
# REBUILD COMPLETE FEATURE MATRIX
# ============================================================

section("RECONSTRUCTING FINAL FEATURE MATRIX")

X_validation = sparse.hstack(
    [
        X_structured,
        *multilabel_matrices,
        X_description,
        X_title,
    ],
    format="csr",
    dtype=np.float32,
)

actual_feature_count = X_validation.shape[1]
expected_feature_count = model.coef_.shape[1]

print("Validation matrix:", X_validation.shape)
print(f"Expected features: {expected_feature_count:,}")
print(f"Reconstructed features: {actual_feature_count:,}")

require(
    actual_feature_count == expected_feature_count,
    "Reconstructed feature count differs from classifier.",
)

require(
    actual_feature_count
    == schema["combined_feature_count"],
    "Feature count differs from saved feature schema.",
)


# ============================================================
# RECOMPUTE VALIDATION METRICS
# ============================================================

section("REPRODUCING VALIDATION METRICS")

scores = model.decision_function(
    X_validation
)

pr_auc = average_precision_score(
    y_validation,
    scores,
)

roc_auc = roc_auc_score(
    y_validation,
    scores,
)

threshold = float(
    bundle["decision_threshold"]
)

predictions = (
    scores >= threshold
).astype(np.int8)

tn, fp, fn, tp = confusion_matrix(
    y_validation,
    predictions,
    labels=[0, 1],
).ravel()

precision = (
    tp / (tp + fp)
    if tp + fp > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if tp + fn > 0
    else 0.0
)

f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0,
)

actual_metrics = {
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "threshold": float(threshold),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),
}

for key, value in actual_metrics.items():
    print(f"{key}: {value}")


# ============================================================
# COMPARE AGAINST PACKAGING REPORT
# ============================================================

section("COMPARING AGAINST SAVED VALIDATION RESULTS")

with METRICS_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    recorded = json.load(file)

# Correct mapping:
# finalize_model.py writes validation_pr_auc and
# validation_roc_auc, not pr_auc and roc_auc.

recorded_metrics = {
    "pr_auc": recorded.get("validation_pr_auc"),
    "roc_auc": recorded.get("validation_roc_auc"),
    "threshold": recorded.get("threshold"),
    "precision": recorded.get("precision"),
    "recall": recorded.get("recall"),
    "f1": recorded.get("f1"),
    "true_positives": recorded.get("true_positives"),
    "false_positives": recorded.get("false_positives"),
    "true_negatives": recorded.get("true_negatives"),
    "false_negatives": recorded.get("false_negatives"),
}

missing_metrics = [
    key
    for key, value in recorded_metrics.items()
    if value is None
]

require(
    not missing_metrics,
    f"Saved report is missing metrics: {missing_metrics}",
)

float_results = {}

for key in [
    "pr_auc",
    "roc_auc",
    "threshold",
    "precision",
    "recall",
    "f1",
]:

    float_results[key] = compare_float(
        key,
        actual_metrics[key],
        recorded_metrics[key],
    )

count_results = {}

for key in [
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
]:

    actual = int(actual_metrics[key])
    expected = int(recorded_metrics[key])

    passed = actual == expected

    count_results[key] = {
        "actual": actual,
        "recorded": expected,
        "passed": passed,
    }

    print(
        f"{key}: actual={actual}, "
        f"recorded={expected}, PASS={passed}"
    )

numeric_checks_passed = all(
    value["passed"]
    for value in float_results.values()
)

count_checks_passed = all(
    value["passed"]
    for value in count_results.values()
)

passed = (
    numeric_checks_passed
    and count_checks_passed
)


# ============================================================
# SAVE VERIFICATION REPORT
# ============================================================

report = {
    "project": "SteamPulse",
    "verification": "saved_model_bundle_reproduction",
    "passed": bool(passed),
    "model_path": str(MODEL_PATH),
    "validation_rows": int(len(validation)),
    "test_partition_used": False,
    "expected_feature_count": int(expected_feature_count),
    "reconstructed_feature_count": int(actual_feature_count),
    "actual_metrics": actual_metrics,
    "recorded_metrics": recorded_metrics,
    "numeric_comparisons": float_results,
    "count_comparisons": count_results,
}

with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        indent=2,
    )


# ============================================================
# FINAL RESULT
# ============================================================

section("VERIFICATION RESULT")

if passed:

    print("PASS: Saved model reproduced recorded validation metrics.")
    print("PASS: Feature dimensions match the classifier.")
    print("PASS: Saved preprocessing components work.")

else:

    print("FAIL: At least one verification check did not match.")
    print("Keep the release candidate unchanged until investigated.")

print(f"\nVerification report: {REPORT_PATH}")
print("No model training was performed.")
print("No test-set evaluation was performed.")

if not passed:
    raise SystemExit(1)

print("\nVerification completed successfully.")