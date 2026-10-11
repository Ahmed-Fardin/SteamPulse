"""
SteamPulse — Final Logistic Regression Hyperparameter Search
============================================================

Purpose:
    Perform a bounded C search around the best current Logistic
    Regression configuration.

Models:
    Logistic Regression with C =
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70

Features:
    - Standardized numerical metadata
    - One-hot categorical metadata
    - Multi-label genres and categories
    - Game-description word TF-IDF
    - Game-title character TF-IDF

Evaluation:
    - Validation PR-AUC
    - Validation ROC-AUC
    - Precision
    - Recall
    - F1 score
    - Convergence
    - Confusion matrix counts

Rules:
    - Fit all preprocessing on training data only.
    - Use validation data for candidate comparison.
    - Never use test data in this experiment.
    - Never modify source datasets.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

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

COMPARISON_PATH = (
    REPORT_DIR / "logistic_regression_final_tuning.csv"
)

THRESHOLD_PATH = (
    REPORT_DIR / "logistic_regression_final_thresholds.csv"
)

# Bounded search around the current best C=0.3.
C_VALUES = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.70,
]

MAX_ITER = 5000
TOLERANCE = 1e-4

MINIMUM_PRECISION = 0.35
TARGET_RECALL = 0.60
TARGET_PR_AUC = 0.45

RANDOM_STATE = 42


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
            f"{dataset_name} is missing required columns: {missing}"
        )


def split_multilabel(value):
    """Parse pipe-separated labels; missing values remain empty."""

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


def choose_threshold(y_true, scores):
    """
    Maximize validation recall subject to precision >= 35%.

    The returned threshold is a decision-score threshold, not
    necessarily a probability threshold.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    # The final precision/recall entries do not correspond to
    # threshold entries.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.flatnonzero(
        precision >= MINIMUM_PRECISION
    )

    if len(eligible) == 0:
        return None

    best_index = eligible[
        np.argmax(recall[eligible])
    ]

    return float(thresholds[best_index])


def evaluate_candidate(C_VALUE, X_train, y_train,
                       X_validation, y_validation):

    print("\n" + "-" * 72)
    print(f"LOGISTIC REGRESSION | C={C_VALUE}")
    print("-" * 72)

    model = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)

        model.fit(X_train, y_train)

    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    converged = len(convergence_warnings) == 0

    scores = model.decision_function(X_validation)

    pr_auc = average_precision_score(
        y_validation,
        scores,
    )

    roc_auc = roc_auc_score(
        y_validation,
        scores,
    )

    threshold = choose_threshold(
        y_validation,
        scores,
    )

    if threshold is None:
        metrics = {
            "threshold": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "true_positives": np.nan,
            "false_positives": np.nan,
            "true_negatives": np.nan,
            "false_negatives": np.nan,
        }

        print("No threshold satisfies the precision requirement.")

    else:
        predictions = (
            scores >= threshold
        ).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision_value = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall_value = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        metrics = {
            "threshold": float(threshold),
            "precision": float(precision_value),
            "recall": float(recall_value),
            "f1": float(
                f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                )
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    iterations = int(np.max(model.n_iter_))

    print(f"Converged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Validation PR-AUC: {pr_auc:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    if not np.isnan(metrics["threshold"]):
        print(f"Threshold: {metrics['threshold']:.6f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"True positives: {metrics['true_positives']}")
        print(f"False positives: {metrics['false_positives']}")

    for warning in convergence_warnings:
        print(f"CONVERGENCE WARNING: {warning.message}")

    return {
        "C": C_VALUE,
        "converged": converged,
        "iterations": iterations,
        "validation_pr_auc": float(pr_auc),
        "validation_roc_auc": float(roc_auc),
        **metrics,
    }


# ================================================================
# LOAD DATA
# ================================================================

section("LOADING STEAMPULSE DATA")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found: {FEATURE_PATH}"
    )

if not TEXT_PATH.exists():
    raise FileNotFoundError(
        f"Text file not found: {TEXT_PATH}"
    )

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
        "Duplicate steam_appid values exist in the text dataset."
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

# Missing outcomes must not be treated as negative outcomes.
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
        "Training or validation partition is empty."
    )

y_train = train_data["hit"].to_numpy()
y_validation = validation_data["hit"].to_numpy()

if len(np.unique(y_train)) != 2:
    raise ValueError(
        "Training data must contain both target classes."
    )

if len(np.unique(y_validation)) != 2:
    raise ValueError(
        "Validation data must contain both target classes."
    )

print(f"Training games: {len(train_data):,}")
print(f"Validation games: {len(validation_data):,}")
print(f"Training HIT prevalence: {y_train.mean():.2%}")
print(f"Validation HIT prevalence: {y_validation.mean():.2%}")


# ================================================================
# STRUCTURED FEATURE PREPROCESSING
# ================================================================

section("PREPARING STRUCTURED FEATURES")

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

preprocessor = ColumnTransformer(
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

X_train_structured = preprocessor.fit_transform(
    train_data[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]
)

X_validation_structured = preprocessor.transform(
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

print("Training structured matrix:", X_train_structured.shape)
print("Validation structured matrix:", X_validation_structured.shape)


# ================================================================
# MULTI-LABEL FEATURES
# ================================================================

section("ENCODING GENRES AND CATEGORIES")

train_multilabel = []
validation_multilabel = []

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

    # New labels in validation cannot introduce fitted features.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        validation_matrix = encoder.transform(
            validation_values
        )

    train_multilabel.append(
        sparse.csr_matrix(
            train_matrix,
            dtype=np.float32,
        )
    )

    validation_multilabel.append(
        sparse.csr_matrix(
            validation_matrix,
            dtype=np.float32,
        )
    )

    print(
        f"  Training label count: {train_matrix.shape[1]:,}"
    )


# ================================================================
# TEXT FEATURES
# ================================================================

section("FITTING TEXT REPRESENTATIONS")

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


print("Fitting description-word TF-IDF...")

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

print(
    f"Description vocabulary: "
    f"{len(description_vectorizer.vocabulary_):,}"
)


print("Fitting title-character TF-IDF...")

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
    f"Title vocabulary: "
    f"{len(title_vectorizer.vocabulary_):,}"
)


# ================================================================
# COMBINE FEATURES
# ================================================================

section("COMBINING FEATURE MATRICES")

X_train = sparse.hstack(
    [
        X_train_structured,
        *train_multilabel,
        X_train_description,
        X_train_title,
    ],
    format="csr",
    dtype=np.float32,
)

X_validation = sparse.hstack(
    [
        X_validation_structured,
        *validation_multilabel,
        X_validation_description,
        X_validation_title,
    ],
    format="csr",
    dtype=np.float32,
)

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_validation.shape)
print(f"Training non-zero values: {X_train.nnz:,}")


# ================================================================
# RUN HYPERPARAMETER SEARCH
# ================================================================

section("LOGISTIC REGRESSION C SEARCH")

results = []

for index, C_VALUE in enumerate(C_VALUES, start=1):

    print(
        f"\nCandidate {index}/{len(C_VALUES)}"
    )

    result = evaluate_candidate(
        C_VALUE=C_VALUE,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    results.append(result)


# ================================================================
# SAVE RESULTS
# ================================================================

section("FINAL COMPARISON")

comparison = pd.DataFrame(results)

# Rank converged candidates by validation PR-AUC.
comparison = comparison.sort_values(
    ["converged", "validation_pr_auc"],
    ascending=[False, False],
).reset_index(drop=True)

comparison.insert(
    0,
    "rank",
    np.arange(1, len(comparison) + 1),
)

comparison.to_csv(
    COMPARISON_PATH,
    index=False,
)

threshold_columns = [
    "C",
    "converged",
    "threshold",
    "precision",
    "recall",
    "f1",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
]

comparison[threshold_columns].to_csv(
    THRESHOLD_PATH,
    index=False,
)

print(
    comparison[
        [
            "rank",
            "C",
            "converged",
            "validation_pr_auc",
            "validation_roc_auc",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(f"\nSaved comparison: {COMPARISON_PATH}")
print(f"Saved thresholds: {THRESHOLD_PATH}")


# ================================================================
# CHECK GOALS
# ================================================================

section("STEAMPULSE TARGET CHECK")

eligible = comparison[
    comparison["converged"]
    & (comparison["validation_pr_auc"] >= TARGET_PR_AUC)
    & (comparison["precision"] >= MINIMUM_PRECISION)
    & (comparison["recall"] >= TARGET_RECALL)
]

if not eligible.empty:

    print("Candidates satisfying all three validation targets:")

    print(
        eligible[
            [
                "C",
                "validation_pr_auc",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )

else:

    converged_models = comparison[
        comparison["converged"]
    ]

    if converged_models.empty:
        print("No candidate converged.")

    else:
        best = converged_models.iloc[0]

        print(
            "No candidate satisfied all three targets."
        )

        print(
            f"Best converged candidate: C={best['C']}"
        )

        print(
            f"PR-AUC: {best['validation_pr_auc']:.4f}"
        )

        print(
            f"Precision: {best['precision']:.2%}"
        )

        print(
            f"Recall: {best['recall']:.2%}"
        )

print("\nNo test data was used.")
print("Experiment completed.")