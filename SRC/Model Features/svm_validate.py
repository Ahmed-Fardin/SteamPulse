"""
SteamPulse — Linear SVM Convergence Experiment
================================================

Purpose:
    1. Investigate Linear SVM convergence.
    2. Standardize numerical structured features.
    3. Combine structured features with game-description and title text.
    4. Compare Linear SVM models with C=0.3 and C=1.0.
    5. Select a validation threshold that maximizes recall while
       maintaining precision >= 0.35.

Important:
    - Uses TRAIN and VALIDATION data only.
    - Does NOT use the test set.
    - Does NOT modify or overwrite source datasets.
    - Fits preprocessing on training data only.
    - Uses tol=0.01 and max_iter=50,000.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# ================================================================
# CONFIGURATION
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT / "Data" / "Processed" / "steam_features_pricefixed.csv"
)

TEXT_PATH = (
    ROOT / "Data" / "Processed" / "steam_modeling_final_prices_fixed.csv"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PATH = REPORT_DIR / "svm_convergence_comparison.csv"
THRESHOLD_PATH = REPORT_DIR / "svm_convergence_thresholds.csv"

C_VALUES = [0.3, 1.0]

MAX_ITER = 50_000
TOLERANCE = 0.01

MINIMUM_PRECISION = 0.35

RANDOM_STATE = 42


# Structured numerical features.
NUMERIC_FEATURES = [
    "release_year",
    "release_month",
    "price_log1p_final",
    "price_missing_final",
    "is_free",
    "required_age",
    "windows",
    "mac",
    "linux",
    "platform_count",
    "genre_count",
    "category_count",
    "developer_count",
    "publisher_count",
    "language_count",
    "has_controller_support",
    "has_pc_requirements",
    "has_mac_requirements",
    "has_linux_requirements",
    "name_length",
    "short_description_length",
    "about_length",
]

# Structured categorical features.
CATEGORICAL_FEATURES = [
    "main_genre",
    "main_category",
    "main_developer",
    "main_publisher",
]

# Multi-label structured features.
MULTILABEL_FEATURES = [
    "genres_multilabel",
    "categories_multilabel",
]

TEXT_REQUIRED_COLUMNS = [
    "steam_appid",
    "name",
    "short_description",
    "about_the_game",
]


# ================================================================
# HELPERS
# ================================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check_columns(df, required, dataset_name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def split_multilabel(value):
    """
    Parse pipe-separated multi-label values.

    Examples:
        'Action|Indie|RPG'
        None
        NaN

    Missing values become empty lists.
    """

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


def select_threshold(y_true, scores, minimum_precision=0.35):
    """
    Select the threshold giving maximum validation recall while
    maintaining the specified minimum validation precision.

    LinearSVC decision scores are NOT probabilities.
    Their thresholds can be negative or positive.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    # precision and recall contain one extra entry compared with
    # thresholds. The final precision/recall entry has no threshold.
    precision_at_thresholds = precision[:-1]
    recall_at_thresholds = recall[:-1]

    eligible = np.flatnonzero(
        precision_at_thresholds >= minimum_precision
    )

    if len(eligible) == 0:
        raise RuntimeError(
            f"No validation threshold achieved precision >= "
            f"{minimum_precision:.2%}. "
            "Inspect the precision-recall curve before proceeding."
        )

    # Find the eligible threshold with maximum recall.
    best_index = eligible[
        np.argmax(recall_at_thresholds[eligible])
    ]

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision_at_thresholds[best_index]),
        "recall": float(recall_at_thresholds[best_index]),
    }


def classification_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


# ================================================================
# LOAD DATA
# ================================================================

section("STEAMPULSE LINEAR SVM CONVERGENCE EXPERIMENT")

print(f"Project root: {ROOT}")
print(f"Maximum iterations: {MAX_ITER:,}")
print(f"Tolerance: {TOLERANCE}")
print(f"C values: {C_VALUES}")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Structured feature file not found:\n{FEATURE_PATH}"
    )

if not TEXT_PATH.exists():
    raise FileNotFoundError(
        f"Text data file not found:\n{TEXT_PATH}"
    )

print("\nLoading structured features...")

features = pd.read_csv(
    FEATURE_PATH,
    low_memory=False,
)

check_columns(
    features,
    [
        "steam_appid",
        "split",
        "hit",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        *MULTILABEL_FEATURES,
    ],
    "Structured features",
)

print(f"Structured dataset shape: {features.shape}")

print("\nLoading game titles and descriptions...")

text_data = pd.read_csv(
    TEXT_PATH,
    usecols=TEXT_REQUIRED_COLUMNS,
    low_memory=False,
)

text_data = text_data.rename(
    columns={"name": "text_name"}
)

# Normalize join keys without modifying the source files.
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

features["steam_appid"] = features["steam_appid"].astype("int64")
text_data["steam_appid"] = text_data["steam_appid"].astype("int64")

if text_data["steam_appid"].duplicated().any():
    raise ValueError(
        "Duplicate steam_appid values found in the text dataset. "
        "Resolve the duplicates before training."
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

# Exclude games whose recommendation outcomes are unobserved.
data = data[
    data["split"].isin(["train", "validation"])
    & data["hit"].notna()
].copy()

data["hit"] = data["hit"].astype(int)

if not set(data["hit"].unique()).issubset({0, 1}):
    raise ValueError(
        "Unexpected label values in 'hit'. Expected only 0 and 1."
    )

train_data = data[data["split"] == "train"].copy()
validation_data = data[data["split"] == "validation"].copy()

if train_data.empty or validation_data.empty:
    raise ValueError(
        "Training or validation data is empty. "
        "Check the split column in the input dataset."
    )

if train_data["hit"].nunique() != 2:
    raise ValueError(
        "Training data must contain both hit classes."
    )

if validation_data["hit"].nunique() != 2:
    raise ValueError(
        "Validation data must contain both hit classes."
    )

y_train = train_data["hit"].to_numpy(dtype=np.int8)
y_validation = validation_data["hit"].to_numpy(dtype=np.int8)

print(f"\nTraining games: {len(train_data):,}")
print(f"Validation games: {len(validation_data):,}")

print(
    f"Training HIT prevalence: "
    f"{y_train.mean():.2%}"
)

print(
    f"Validation HIT prevalence: "
    f"{y_validation.mean():.2%}"
)

# This script deliberately does not select, load, or evaluate test rows.


# ================================================================
# PREPARE STRUCTURED FEATURES
# ================================================================

section("PREPROCESSING STRUCTURED FEATURES")

STRUCTURED_FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

X_train_structured = train_data[STRUCTURED_FEATURES].copy()
X_validation_structured = validation_data[
    STRUCTURED_FEATURES
].copy()

# Numerical values are imputed and standardized.
#
# Standardization makes numerical feature scales more comparable.
# This is especially relevant when combining numerical metadata
# with sparse TF-IDF features.

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

# Pool infrequent categories to prevent developer/publisher
# columns from growing unnecessarily large.
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                max_categories=200,
            ),
        ),
    ]
)

structured_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    sparse_threshold=1.0,
)

print("Fitting structured preprocessing on training data...")

X_train_structured_matrix = (
    structured_preprocessor.fit_transform(
        X_train_structured
    )
)

X_validation_structured_matrix = (
    structured_preprocessor.transform(
        X_validation_structured
    )
)

X_train_structured_matrix = sparse.csr_matrix(
    X_train_structured_matrix,
    dtype=np.float32,
)

X_validation_structured_matrix = sparse.csr_matrix(
    X_validation_structured_matrix,
    dtype=np.float32,
)

print(
    "Training structured matrix:",
    X_train_structured_matrix.shape,
)

print(
    "Validation structured matrix:",
    X_validation_structured_matrix.shape,
)


# ================================================================
# MULTI-LABEL FEATURES
# ================================================================

section("PREPARING MULTI-LABEL FEATURES")

train_multilabel_matrices = []
validation_multilabel_matrices = []

for column in MULTILABEL_FEATURES:

    print(f"Processing {column}...")

    train_values = (
        train_data[column]
        .apply(split_multilabel)
        .tolist()
    )

    validation_values = (
        validation_data[column]
        .apply(split_multilabel)
        .tolist()
    )

    binarizer = MultiLabelBinarizer(
        sparse_output=True
    )

    train_matrix = binarizer.fit_transform(
        train_values
    )

    validation_matrix = binarizer.transform(
        validation_values
    )

    train_matrix = sparse.csr_matrix(
        train_matrix,
        dtype=np.float32,
    )

    validation_matrix = sparse.csr_matrix(
        validation_matrix,
        dtype=np.float32,
    )

    train_multilabel_matrices.append(train_matrix)
    validation_multilabel_matrices.append(validation_matrix)

    print(
        f"  Training labels: {train_matrix.shape[1]:,}"
    )

    print(
        f"  Validation shape: {validation_matrix.shape}"
    )


# ================================================================
# TEXT FEATURES
# ================================================================

section("FITTING TEXT REPRESENTATIONS")

# Missing descriptions become empty strings.
#
# Word TF-IDF uses descriptions.
# Character TF-IDF uses titles.
#
# Text vocabularies are fitted exclusively on training games.

train_descriptions = (
    train_data["short_description"]
    .fillna("")
    .astype(str)
    + " "
    + train_data["about_the_game"]
    .fillna("")
    .astype(str)
).tolist()

validation_descriptions = (
    validation_data["short_description"]
    .fillna("")
    .astype(str)
    + " "
    + validation_data["about_the_game"]
    .fillna("")
    .astype(str)
).tolist()

train_titles = (
    train_data["text_name"]
    .fillna("")
    .astype(str)
).tolist()

validation_titles = (
    validation_data["text_name"]
    .fillna("")
    .astype(str)
).tolist()


print("Fitting description-word TF-IDF...")

description_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_descriptions = description_vectorizer.fit_transform(
    train_descriptions
)

X_validation_descriptions = description_vectorizer.transform(
    validation_descriptions
)

print(
    "Description vocabulary:",
    len(description_vectorizer.vocabulary_),
)


print("Fitting title character n-grams...")

title_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=15_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_titles = title_vectorizer.fit_transform(
    train_titles
)

X_validation_titles = title_vectorizer.transform(
    validation_titles
)

print(
    "Title character vocabulary:",
    len(title_vectorizer.vocabulary_),
)


# ================================================================
# COMBINE FEATURES
# ================================================================

section("COMBINING FEATURE MATRICES")

train_matrices = [
    X_train_structured_matrix,
    *train_multilabel_matrices,
    X_train_descriptions,
    X_train_titles,
]

validation_matrices = [
    X_validation_structured_matrix,
    *validation_multilabel_matrices,
    X_validation_descriptions,
    X_validation_titles,
]

X_train = sparse.hstack(
    train_matrices,
    format="csr",
    dtype=np.float32,
)

X_validation = sparse.hstack(
    validation_matrices,
    format="csr",
    dtype=np.float32,
)

print("Combined training matrix:", X_train.shape)
print("Combined validation matrix:", X_validation.shape)

print(f"Training matrix non-zero values: {X_train.nnz:,}")

print(
    f"Training matrix density: "
    f"{X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.6%}"
)


# ================================================================
# TRAIN AND EVALUATE LINEAR SVM
# ================================================================

section("LINEAR SVM CONVERGENCE CHECK")

comparison_rows = []
threshold_rows = []

for index, C_VALUE in enumerate(C_VALUES, start=1):

    print("\n" + "-" * 68)
    print(f"LINEAR SVM {index}/{len(C_VALUES)} | C={C_VALUE}")
    print("-" * 68)

    model = LinearSVC(
        C=C_VALUE,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        class_weight="balanced",
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        random_state=RANDOM_STATE,
    )

    convergence_warnings = []

    with warnings.catch_warnings(record=True) as caught_warnings:

        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            X_train,
            y_train,
        )

        convergence_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(
                warning.category,
                ConvergenceWarning,
            )
        ]

    converged = len(convergence_warnings) == 0

    iterations_used = int(model.n_iter_)

    validation_scores = model.decision_function(
        X_validation
    )

    validation_ap = average_precision_score(
        y_validation,
        validation_scores,
    )

    validation_roc = roc_auc_score(
        y_validation,
        validation_scores,
    )

    threshold_result = select_threshold(
        y_true=y_validation,
        scores=validation_scores,
        minimum_precision=MINIMUM_PRECISION,
    )

    metrics = classification_metrics(
        y_true=y_validation,
        scores=validation_scores,
        threshold=threshold_result["threshold"],
    )

    print(f"Converged: {converged}")
    print(f"Iterations used: {iterations_used:,}")

    if converged:
        print("Convergence warning: None")
    else:
        print(
            "WARNING: Solver did not converge. "
            "Treat these results as provisional."
        )

    print(f"Validation PR-AUC: {validation_ap:.4f}")
    print(f"Validation ROC-AUC: {validation_roc:.4f}")

    print(
        f"Threshold: {metrics['threshold']:.6f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(f"F1: {metrics['f1']:.4f}")

    print(
        f"True positives: {metrics['true_positives']}"
    )

    print(
        f"False positives: {metrics['false_positives']}"
    )

    print(
        f"False negatives: {metrics['false_negatives']}"
    )

    print(
        f"True negatives: {metrics['true_negatives']}"
    )

    if convergence_warnings:
        for warning in convergence_warnings:
            print(f"Solver warning: {warning.message}")

    comparison_rows.append(
        {
            "C": C_VALUE,
            "converged": converged,
            "iterations": iterations_used,
            "tolerance": TOLERANCE,
            "max_iter": MAX_ITER,
            "validation_pr_auc": validation_ap,
            "validation_roc_auc": validation_roc,
            **metrics,
        }
    )

    threshold_rows.append(
        {
            "C": C_VALUE,
            "threshold": metrics["threshold"],
            "minimum_precision_target": MINIMUM_PRECISION,
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "validation_f1": metrics["f1"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "true_negatives": metrics["true_negatives"],
            "false_negatives": metrics["false_negatives"],
            "converged": converged,
        }
    )


# ================================================================
# SAVE REPORTS
# ================================================================

section("SAVING RESULTS")

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    "validation_pr_auc",
    ascending=False,
).reset_index(drop=True)

comparison_df.insert(
    0,
    "rank",
    np.arange(1, len(comparison_df) + 1),
)

threshold_df = pd.DataFrame(threshold_rows)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)

threshold_df.to_csv(
    THRESHOLD_PATH,
    index=False,
)

print("\nFinal comparison:")
print(comparison_df.to_string(index=False))

print(f"\nSaved comparison: {COMPARISON_PATH}")
print(f"Saved threshold report: {THRESHOLD_PATH}")

print("\nTarget checklist:")

for _, row in comparison_df.iterrows():

    precision_ok = (
        row["precision"] >= MINIMUM_PRECISION
    )

    recall_ok = row["recall"] >= 0.60

    ap_ok = row["validation_pr_auc"] >= 0.45

    print(
        f"C={row['C']}: "
        f"converged={row['converged']}, "
        f"PR-AUC target={ap_ok}, "
        f"precision target={precision_ok}, "
        f"recall target={recall_ok}"
    )

print("\nNo test data was used.")
print("Experiment completed.")