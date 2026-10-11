"""
SteamPulse — Controlled Linear Model Comparison
================================================

Compare Logistic Regression and Linear SVM using the same:

    - Training and validation observations
    - Structured feature preprocessing
    - Multi-label feature representation
    - Description-word TF-IDF
    - Title character TF-IDF

Models:
    Logistic Regression: C = 0.1, 0.3, 1.0
    Linear SVM:          C = 0.3, 1.0

Evaluation:
    - Validation PR-AUC
    - Validation ROC-AUC
    - Precision
    - Recall
    - F1 score
    - Confusion matrix counts
    - Convergence status

No test data is used.
No source dataset is modified.
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
from sklearn.svm import LinearSVC


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
    REPORT_DIR / "linear_model_benchmark.csv"
)

THRESHOLD_PATH = (
    REPORT_DIR / "linear_model_benchmark_thresholds.csv"
)

MINIMUM_PRECISION = 0.35
TARGET_RECALL = 0.60
TARGET_PR_AUC = 0.45

RANDOM_STATE = 42

# Logistic Regression settings.
LOGISTIC_C_VALUES = [0.1, 0.3, 1.0]
LOGISTIC_MAX_ITER = 5000
LOGISTIC_TOL = 1e-4

# Linear SVM settings.
SVM_C_VALUES = [0.3, 1.0]
SVM_MAX_ITER = 50000
SVM_TOL = 0.01


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
# HELPER FUNCTIONS
# ================================================================

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
    """Parse pipe-separated labels without inventing missing labels."""

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
    Find the threshold with maximum validation recall subject to
    validation precision >= MINIMUM_PRECISION.

    SVM decision scores and Logistic Regression decision scores
    are not constrained to the interval [0, 1].
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

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


def evaluate_model(model_name, C_value, model, X_train, y_train,
                   X_validation, y_validation):

    print("\n" + "-" * 72)
    print(f"MODEL: {model_name} | C={C_value}")
    print("-" * 72)

    caught_convergence_warnings = []

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always", ConvergenceWarning)

        model.fit(X_train, y_train)

        caught_convergence_warnings = [
            warning
            for warning in caught
            if issubclass(
                warning.category,
                ConvergenceWarning,
            )
        ]

    converged = len(caught_convergence_warnings) == 0

    scores = model.decision_function(X_validation)

    ap = average_precision_score(
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

        print(
            "No threshold achieved the minimum precision "
            f"of {MINIMUM_PRECISION:.0%}."
        )

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

    iterations = getattr(model, "n_iter_", None)

    if iterations is not None:
        iterations = int(np.max(iterations))

    print(f"Converged: {converged}")

    if iterations is not None:
        print(f"Iterations used: {iterations:,}")

    print(f"Validation PR-AUC: {ap:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    if not np.isnan(metrics["threshold"]):
        print(f"Threshold: {metrics['threshold']:.6f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"True positives: {metrics['true_positives']}")
        print(f"False positives: {metrics['false_positives']}")

    if caught_convergence_warnings:
        for warning in caught_convergence_warnings:
            print(f"WARNING: {warning.message}")

    result = {
        "model": model_name,
        "C": C_value,
        "converged": converged,
        "iterations": iterations,
        "validation_pr_auc": float(ap),
        "validation_roc_auc": float(roc_auc),
        **metrics,
    }

    return result


# ================================================================
# LOAD AND VALIDATE DATA
# ================================================================

section("LOADING DATA")

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
    "Structured features",
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
        "Duplicate steam_appid values found in the text dataset."
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

# Only observed outcomes can be used for supervised training.
data = data[
    data["split"].isin(["train", "validation"])
    & data["hit"].notna()
].copy()

data["hit"] = data["hit"].astype("int8")

train_data = data[
    data["split"] == "train"
].copy()

validation_data = data[
    data["split"] == "validation"
].copy()

if train_data.empty or validation_data.empty:
    raise ValueError(
        "The train or validation partition is empty."
    )

y_train = train_data["hit"].to_numpy()
y_validation = validation_data["hit"].to_numpy()

if len(np.unique(y_train)) != 2:
    raise ValueError("Training labels must contain both classes.")

if len(np.unique(y_validation)) != 2:
    raise ValueError("Validation labels must contain both classes.")

print(f"Training games: {len(train_data):,}")
print(f"Validation games: {len(validation_data):,}")
print(f"Training HIT prevalence: {y_train.mean():.2%}")
print(f"Validation HIT prevalence: {y_validation.mean():.2%}")


# ================================================================
# STRUCTURED FEATURES
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

print("Training structured shape:", X_train_structured.shape)
print("Validation structured shape:", X_validation_structured.shape)


# ================================================================
# MULTI-LABEL FEATURES
# ================================================================

section("PREPARING MULTI-LABEL FEATURES")

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

    # Classes absent from training are intentionally not learned
    # from validation data. They cannot become fitted predictors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        validation_matrix = encoder.transform(
            validation_values
        )

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
        f"  Training label count: {train_matrix.shape[1]:,}"
    )


# ================================================================
# TEXT FEATURES
# ================================================================

section("PREPARING TEXT FEATURES")

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
# FINAL FEATURE MATRIX
# ================================================================

section("COMBINING FEATURES")

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
print(f"Training non-zero values: {X_train.nnz:,}")


# ================================================================
# MODEL COMPARISON
# ================================================================

section("COMPARING LOGISTIC REGRESSION AND LINEAR SVM")

results = []

# Logistic Regression candidates.
for C_VALUE in LOGISTIC_C_VALUES:

    model = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
        random_state=RANDOM_STATE,
    )

    results.append(
        evaluate_model(
            model_name="Logistic Regression",
            C_value=C_VALUE,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    )


# Linear SVM candidates.
for C_VALUE in SVM_C_VALUES:

    model = LinearSVC(
        C=C_VALUE,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        class_weight="balanced",
        max_iter=SVM_MAX_ITER,
        tol=SVM_TOL,
        random_state=RANDOM_STATE,
    )

    results.append(
        evaluate_model(
            model_name="Linear SVM",
            C_value=C_VALUE,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    )


# ================================================================
# REPORTING
# ================================================================

section("FINAL MODEL COMPARISON")

comparison = pd.DataFrame(results)

# Candidates that failed to converge remain visible in the report,
# but converged models are ranked ahead of non-converged models.
comparison = comparison.sort_values(
    by=[
        "converged",
        "validation_pr_auc",
    ],
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
    "model",
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

display_columns = [
    "rank",
    "model",
    "C",
    "converged",
    "validation_pr_auc",
    "validation_roc_auc",
    "precision",
    "recall",
    "f1",
]

print(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(f"\nComparison saved to: {COMPARISON_PATH}")
print(f"Thresholds saved to: {THRESHOLD_PATH}")


section("VALIDATION TARGET CHECK")

eligible = comparison[
    comparison["converged"]
    & (comparison["validation_pr_auc"] >= TARGET_PR_AUC)
    & (comparison["precision"] >= MINIMUM_PRECISION)
    & (comparison["recall"] >= TARGET_RECALL)
]

if eligible.empty:

    converged_models = comparison[
        comparison["converged"]
    ]

    if converged_models.empty:
        print(
            "No model converged. We should investigate the "
            "optimization configuration before selecting a model."
        )
    else:
        best = converged_models.iloc[0]

        print(
            "No converged candidate achieved every target."
        )

        print(
            f"Best converged candidate by PR-AUC: "
            f"{best['model']} (C={best['C']})"
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

else:
    print(
        "Candidates meeting all three targets:"
    )

    print(
        eligible[
            ["model", "C", "validation_pr_auc", "precision", "recall"]
        ].to_string(index=False)
    )

print("\nNo test data was used.")
print("Experiment completed.")