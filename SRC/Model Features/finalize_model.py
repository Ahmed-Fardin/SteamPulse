"""
SteamPulse — Final Model Packaging
===================================

Creates a reproducible release-candidate artifact containing:

    - Logistic Regression classifier
    - Structured feature preprocessor
    - Multi-label encoders
    - Description TF-IDF vectorizer
    - Title character TF-IDF vectorizer
    - Validation-selected decision threshold
    - Validation metrics
    - Feature schema and target metadata

Selected configuration:
    Logistic Regression, C=0.50

Data partitions:
    Train:      model fitting
    Validation: model evaluation and threshold selection
    Test:       NOT USED

Safety:
    - Existing datasets are read-only.
    - Existing model artifacts are not overwritten.
    - A new release-candidate file is created.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)


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

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Preserve the existing folder's capitalization on Windows.
MODEL_DIR = next(
    (
        path
        for path in ROOT.iterdir()
        if path.is_dir()
        and path.name.lower() == "models"
    ),
    ROOT / "Models",
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / "steampulse_release_candidate.joblib"

METRICS_PATH = (
    REPORT_DIR / "steampulse_release_candidate_validation.json"
)

FEATURE_SCHEMA_PATH = (
    REPORT_DIR / "steampulse_release_candidate_features.json"
)

THRESHOLD_REPORT_PATH = (
    REPORT_DIR / "steampulse_release_candidate_threshold.csv"
)


# Hyperparameters selected from the completed search.
C_VALUE = 0.50
MAX_ITER = 5000
TOLERANCE = 1e-4
RANDOM_STATE = 42

MINIMUM_PRECISION = 0.35


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

CATEGORICAL_FEATURES = [
    "main_genre",
    "main_category",
    "main_developer",
    "main_publisher",
]

MULTILABEL_FEATURES = [
    "genres_multilabel",
    "categories_multilabel",
]

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


def check_columns(df, required, dataset_name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


def split_multilabel(value):
    """Parse pipe-separated labels."""

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


def select_threshold(y_true, scores):
    """
    Maximize validation recall subject to precision >= 35%.

    The threshold is applied to model decision scores, not probabilities.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    # The last precision/recall entry has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.flatnonzero(
        precision >= MINIMUM_PRECISION
    )

    if len(eligible) == 0:
        raise RuntimeError(
            "No threshold meets the minimum precision requirement."
        )

    best_index = eligible[
        np.argmax(recall[eligible])
    ]

    return {
        "threshold": float(thresholds[best_index]),
        "precision_from_curve": float(precision[best_index]),
        "recall_from_curve": float(recall[best_index]),
    }


def calculate_metrics(y_true, scores, threshold):
    predictions = (
        scores >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
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


# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

section("STEAMPULSE MODEL PACKAGING")

print(f"Project root: {ROOT}")
print(f"Model directory: {MODEL_DIR}")
print(f"Output model: {MODEL_PATH}")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {FEATURE_PATH}"
    )

if not TEXT_PATH.exists():
    raise FileNotFoundError(
        f"Text dataset not found: {TEXT_PATH}"
    )

# Never silently overwrite a previous artifact.
if MODEL_PATH.exists():
    raise FileExistsError(
        f"The output artifact already exists:\n{MODEL_PATH}\n"
        "Preserve it. Change MODEL_PATH to a new filename before "
        "creating another artifact."
    )


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING DATA")

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
    "Structured feature dataset",
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

if text_data["steam_appid"].duplicated().any():
    raise ValueError(
        "Duplicate app IDs found in the text dataset."
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

# Only games with observed outcomes can be used in this experiment.
# Games without observed outcomes are not considered negatives.
data = data[
    data["split"].isin(["train", "validation"])
    & data["hit"].notna()
].copy()

data["hit"] = data["hit"].astype(np.int8)

train_data = data[
    data["split"] == "train"
].copy()

validation_data = data[
    data["split"] == "validation"
].copy()

if train_data.empty or validation_data.empty:
    raise ValueError(
        "Training or validation data is empty."
    )

y_train = train_data["hit"].to_numpy()
y_validation = validation_data["hit"].to_numpy()

if len(np.unique(y_train)) != 2:
    raise ValueError(
        "The training partition must contain both classes."
    )

if len(np.unique(y_validation)) != 2:
    raise ValueError(
        "The validation partition must contain both classes."
    )

print(f"Training games: {len(train_data):,}")
print(f"Validation games: {len(validation_data):,}")

print(
    f"Training HIT prevalence: {y_train.mean():.2%}"
)

print(
    f"Validation HIT prevalence: {y_validation.mean():.2%}"
)


# ============================================================
# STRUCTURED PREPROCESSING
# ============================================================

section("FITTING STRUCTURED PREPROCESSOR")

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

X_train_structured = structured_preprocessor.fit_transform(
    train_data[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]
)

X_validation_structured = structured_preprocessor.transform(
    validation_data[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]
)

X_train_structured = sparse.csr_matrix(
    X_train_structured,
    dtype=np.float32,
)

X_validation_structured = sparse.csr_matrix(
    X_validation_structured,
    dtype=np.float32,
)

print(
    "Structured training matrix:",
    X_train_structured.shape,
)


# ============================================================
# MULTI-LABEL PREPROCESSING
# ============================================================

section("FITTING MULTI-LABEL ENCODERS")

multilabel_encoders = {}

train_multilabel_matrices = []
validation_multilabel_matrices = []

for column in MULTILABEL_FEATURES:

    print(f"Encoding {column}...")

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

    encoder = MultiLabelBinarizer(
        sparse_output=True
    )

    train_matrix = encoder.fit_transform(
        train_values
    )

    # Validation labels absent from training are not fitted as features.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        validation_matrix = encoder.transform(
            validation_values
        )

    multilabel_encoders[column] = encoder

    train_multilabel_matrices.append(
        sparse.csr_matrix(
            train_matrix,
            dtype=np.float32,
        )
    )

    validation_multilabel_matrices.append(
        sparse.csr_matrix(
            validation_matrix,
            dtype=np.float32,
        )
    )

    print(
        f"  Fitted training labels: "
        f"{train_matrix.shape[1]:,}"
    )


# ============================================================
# TEXT VECTORIZERS
# ============================================================

section("FITTING TEXT VECTORIZERS")

train_descriptions = (
    train_data["short_description"].fillna("").astype(str)
    + " "
    + train_data["about_the_game"].fillna("").astype(str)
).tolist()

validation_descriptions = (
    validation_data["short_description"].fillna("").astype(str)
    + " "
    + validation_data["about_the_game"].fillna("").astype(str)
).tolist()

train_titles = (
    train_data["text_name"].fillna("").astype(str)
).tolist()

validation_titles = (
    validation_data["text_name"].fillna("").astype(str)
).tolist()


description_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_description = (
    description_vectorizer.fit_transform(
        train_descriptions
    )
)

X_validation_description = (
    description_vectorizer.transform(
        validation_descriptions
    )
)


title_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=15_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_title = title_vectorizer.fit_transform(
    train_titles
)

X_validation_title = title_vectorizer.transform(
    validation_titles
)

print(
    f"Description vocabulary: "
    f"{len(description_vectorizer.vocabulary_):,}"
)

print(
    f"Title vocabulary: "
    f"{len(title_vectorizer.vocabulary_):,}"
)


# ============================================================
# BUILD TRAINING AND VALIDATION MATRICES
# ============================================================

section("BUILDING FINAL FEATURE MATRICES")

X_train = sparse.hstack(
    [
        X_train_structured,
        *train_multilabel_matrices,
        X_train_description,
        X_train_title,
    ],
    format="csr",
    dtype=np.float32,
)

X_validation = sparse.hstack(
    [
        X_validation_structured,
        *validation_multilabel_matrices,
        X_validation_description,
        X_validation_title,
    ],
    format="csr",
    dtype=np.float32,
)

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_validation.shape)

if X_train.shape[1] != X_validation.shape[1]:
    raise RuntimeError(
        "Training and validation feature dimensions do not match."
    )


# ============================================================
# TRAIN SELECTED MODEL
# ============================================================

section("TRAINING SELECTED LOGISTIC REGRESSION MODEL")

model = LogisticRegression(
    C=C_VALUE,
    penalty="l2",
    solver="liblinear",
    class_weight="balanced",
    max_iter=MAX_ITER,
    tol=TOLERANCE,
    random_state=RANDOM_STATE,
)

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

print(f"Converged: {converged}")
print(f"Iterations: {int(np.max(model.n_iter_)):,}")

if not converged:
    for warning in convergence_warnings:
        print(f"WARNING: {warning.message}")

    raise RuntimeError(
        "The final Logistic Regression model did not converge. "
        "No release-candidate model will be saved."
    )


# ============================================================
# VALIDATION EVALUATION AND THRESHOLD SELECTION
# ============================================================

section("EVALUATING ON VALIDATION DATA")

validation_scores = model.decision_function(
    X_validation
)

validation_pr_auc = average_precision_score(
    y_validation,
    validation_scores,
)

validation_roc_auc = roc_auc_score(
    y_validation,
    validation_scores,
)

threshold_info = select_threshold(
    y_validation,
    validation_scores,
)

threshold = threshold_info["threshold"]

validation_metrics = calculate_metrics(
    y_validation,
    validation_scores,
    threshold,
)

print(f"Validation PR-AUC: {validation_pr_auc:.4f}")
print(f"Validation ROC-AUC: {validation_roc_auc:.4f}")
print(f"Selected threshold: {threshold:.6f}")
print(f"Precision: {validation_metrics['precision']:.4f}")
print(f"Recall: {validation_metrics['recall']:.4f}")
print(f"F1: {validation_metrics['f1']:.4f}")
print(
    f"True positives: "
    f"{validation_metrics['true_positives']}"
)
print(
    f"False positives: "
    f"{validation_metrics['false_positives']}"
)


# ============================================================
# CREATE MODEL BUNDLE
# ============================================================

section("CREATING MODEL BUNDLE")

bundle = {
    "project": "SteamPulse",

    "artifact_type": "release_candidate",

    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "model_name": "Logistic Regression",

    "model": model,

    "hyperparameters": {
        "C": C_VALUE,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": MAX_ITER,
        "tol": TOLERANCE,
        "random_state": RANDOM_STATE,
    },

    "decision_threshold": threshold,

    "score_type": "decision_function",

    "threshold_selection": {
        "method": (
            "Maximum validation recall subject to "
            "validation precision >= 0.35"
        ),
        "minimum_precision": MINIMUM_PRECISION,
        **threshold_info,
    },

    "preprocessing": {
        "structured_preprocessor": structured_preprocessor,

        "multilabel_encoders": multilabel_encoders,

        "description_vectorizer": description_vectorizer,

        "title_vectorizer": title_vectorizer,
    },

    "feature_schema": {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "multilabel_features": MULTILABEL_FEATURES,
        "text_columns": TEXT_COLUMNS,
        "combined_feature_count": int(X_train.shape[1]),
    },

    "target_definition": {
        "name": "hit",
        "positive_class": 1,
        "negative_class": 0,
        "description": (
            "Binary label based on the defined Steam "
            "recommendation threshold. Missing outcomes "
            "are excluded from supervised training."
        ),
        "interpretation_warning": (
            "The target is a proxy for cumulative observed "
            "recommendation engagement, not verified commercial "
            "success. Current metadata is not necessarily "
            "metadata available at release."
        ),
    },

    "data_partitions": {
        "training_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)),
        "test_used": False,
    },

    "validation_metrics": {
        "pr_auc": float(validation_pr_auc),
        "roc_auc": float(validation_roc_auc),
        **validation_metrics,
    },

    "software_versions": {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    },
}


# ============================================================
# VERIFY SHAPES AND SAVE ARTIFACT
# ============================================================

section("VERIFYING FINAL ARTIFACT")

if not (
    len(NUMERIC_FEATURES)
    and len(CATEGORICAL_FEATURES)
    and len(MULTILABEL_FEATURES)
):
    raise RuntimeError(
        "Unexpected empty feature configuration."
    )

if bundle["feature_schema"]["combined_feature_count"] != (
    X_validation.shape[1]
):
    raise RuntimeError(
        "Feature schema dimension check failed."
    )

# Save the complete model bundle to a NEW filename.
joblib.dump(
    bundle,
    MODEL_PATH,
    compress=3,
)

# Save validation metrics in JSON.
with METRICS_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "project": "SteamPulse",
            "model_name": "Logistic Regression",
            "C": C_VALUE,
            "converged": converged,
            "training_rows": int(len(train_data)),
            "validation_rows": int(len(validation_data)),
            "validation_pr_auc": float(validation_pr_auc),
            "validation_roc_auc": float(validation_roc_auc),
            **validation_metrics,
            "test_used": False,
        },
        file,
        indent=2,
    )

# Save a machine-readable feature schema.
with FEATURE_SCHEMA_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        bundle["feature_schema"],
        file,
        indent=2,
    )

# Save threshold details separately for quick inspection.
pd.DataFrame(
    [
        {
            "model": "Logistic Regression",
            "C": C_VALUE,
            "threshold": threshold,
            "score_type": "decision_function",
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_pr_auc": float(validation_pr_auc),
            "validation_roc_auc": float(validation_roc_auc),
        }
    ]
).to_csv(
    THRESHOLD_REPORT_PATH,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

section("MODEL PACKAGING COMPLETE")

print(f"Model artifact: {MODEL_PATH}")
print(f"Validation metrics: {METRICS_PATH}")
print(f"Feature schema: {FEATURE_SCHEMA_PATH}")
print(f"Threshold report: {THRESHOLD_REPORT_PATH}")

print("\nFinal validation results:")
print(f"  PR-AUC: {validation_pr_auc:.4f}")
print(f"  ROC-AUC: {validation_roc_auc:.4f}")
print(f"  Precision: {validation_metrics['precision']:.2%}")
print(f"  Recall: {validation_metrics['recall']:.2%}")
print(f"  Threshold: {threshold:.6f}")

print("\nThe test set was not used.")
print("Existing model artifacts were not overwritten.")
print("Model packaging completed.")