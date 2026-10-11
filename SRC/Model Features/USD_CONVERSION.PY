from pathlib import Path
import inspect

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "Data/Processed/steam_features_pricefixed.csv"
)

COMPARISON_FILE = Path(
    "reports/price_ablation_comparison.csv"
)

THRESHOLD_FILE = Path(
    "reports/price_ablation_thresholds.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

EXPECTED_COUNTS = {
    "train": 13369,
    "validation": 4372,
    "test": 4884,
}

# Minimum validation precision for our initial
# high-recall operating point.
MIN_PRECISION = 0.35


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


# ============================================================
# PREPROCESSING
# ============================================================

def split_labels(value):
    """Convert pipe-separated genre/category strings into labels."""

    if not isinstance(value, str) or not value.strip():
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def make_one_hot_encoder():
    """Support different scikit-learn versions."""

    parameters = inspect.signature(
        OneHotEncoder
    ).parameters

    kwargs = {
        "handle_unknown": "ignore",
    }

    if "min_frequency" in parameters:
        kwargs["min_frequency"] = 5

    return OneHotEncoder(**kwargs)


def make_preprocessor(numeric_features):
    """Build preprocessing fitted only on training data."""

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(with_mean=False),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "encoder",
            make_one_hot_encoder(),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "genres",
                CountVectorizer(
                    tokenizer=split_labels,
                    token_pattern=None,
                    lowercase=False,
                    binary=True,
                    min_df=5,
                ),
                "genres_multilabel",
            ),
            (
                "categories",
                CountVectorizer(
                    tokenizer=split_labels,
                    token_pattern=None,
                    lowercase=False,
                    binary=True,
                    min_df=5,
                ),
                "categories_multilabel",
            ),
        ],
        remainder="drop",
    )


def prepare_features(df, numeric_features):
    """Select features and standardize their input types."""

    columns = (
        numeric_features
        + CATEGORICAL_FEATURES
        + MULTILABEL_FEATURES
    )

    X = df[columns].copy()

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
        )

    for column in MULTILABEL_FEATURES:
        X[column] = (
            X[column]
            .fillna("")
            .astype(str)
        )

    return X


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold(y_true, probabilities):
    """
    Maximize validation recall while maintaining at least
    MIN_PRECISION. If none qualify, fall back to maximum F1.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No threshold candidates were produced."
        )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    eligible = table[
        table["precision"] >= MIN_PRECISION
    ]

    if not eligible.empty:
        selected = eligible.sort_values(
            ["recall", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

        reason = "maximum_recall_with_precision_floor"

    else:
        selected = table.sort_values(
            ["f1", "precision"],
            ascending=[False, False],
        ).iloc[0]

        reason = "maximum_f1_fallback"

    return float(selected["threshold"]), table, reason


def threshold_metrics(y_true, probabilities, threshold):
    """Calculate precision, recall and confusion counts."""

    y_true = np.asarray(y_true)
    predicted = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    tp = int(((y_true == 1) & (predicted == 1)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())
    tn = int(((y_true == 0) & (predicted == 0)).sum())

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

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "predicted_hit_count": tp + fp,
    }


# ============================================================
# MODEL CREATION
# ============================================================

def build_model(model_type, numeric_features):
    """Create a candidate with fixed, reproducible settings."""

    preprocessor = make_preprocessor(
        numeric_features
    )

    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            max_features=0.5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    elif model_type == "logistic_regression":
        estimator = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("STEAMPULSE PRICE ABLATION EXPERIMENT")
    print("=" * 65)

    print("\nLoading corrected feature dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + MULTILABEL_FEATURES
        + [
            "steam_appid",
            "hit",
            "split",
        ]
    )

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}"
        )

    if df["steam_appid"].isna().any():
        raise ValueError(
            "Missing Steam app IDs detected."
        )

    if df["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate Steam app IDs detected."
        )

    df["hit"] = pd.to_numeric(
        df["hit"],
        errors="coerce",
    )

    labeled = df[
        df["hit"].isin([0, 1])
    ].copy()

    train = labeled[
        labeled["split"] == "train"
    ].copy()

    validation = labeled[
        labeled["split"] == "validation"
    ].copy()

    # The test data is not used in this experiment.
    print(f"\nTraining games: {len(train):,}")
    print(f"Validation games: {len(validation):,}")

    for split_name, expected in EXPECTED_COUNTS.items():
        if split_name == "test":
            continue

        actual = int(
            (labeled["split"] == split_name).sum()
        )

        if actual != expected:
            raise ValueError(
                f"{split_name} count changed: "
                f"expected {expected:,}, got {actual:,}."
            )

    y_train = train["hit"].astype(int)
    y_validation = validation["hit"].astype(int)

    # --------------------------------------------------------
    # DEFINE THREE CONTROLLED EXPERIMENTS
    # --------------------------------------------------------

    no_price_features = [
        column
        for column in NUMERIC_FEATURES
        if column != "price_log1p_final"
    ]

    experiments = {
        "Random Forest - corrected price": {
            "model_type": "random_forest",
            "numeric_features": NUMERIC_FEATURES,
        },
        "Random Forest - no monetary price": {
            "model_type": "random_forest",
            "numeric_features": no_price_features,
        },
        "Logistic Regression - corrected price": {
            "model_type": "logistic_regression",
            "numeric_features": NUMERIC_FEATURES,
        },
    }

    comparison_rows = []
    threshold_rows = []

    # --------------------------------------------------------
    # TRAIN AND EVALUATE ON VALIDATION
    # --------------------------------------------------------

    for index, (name, settings) in enumerate(
        experiments.items(),
        start=1,
    ):

        print("\n" + "-" * 65)
        print(f"EXPERIMENT {index}/3: {name}")
        print("-" * 65)

        numeric_features = settings["numeric_features"]

        X_train = prepare_features(
            train,
            numeric_features,
        )

        X_validation = prepare_features(
            validation,
            numeric_features,
        )

        model = build_model(
            settings["model_type"],
            numeric_features,
        )

        print("Training model...")

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        roc_auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        threshold, threshold_table, reason = (
            choose_threshold(
                y_validation,
                probabilities,
            )
        )

        metrics = threshold_metrics(
            y_validation,
            probabilities,
            threshold,
        )

        print(f"Validation PR-AUC: {pr_auc:.4f}")
        print(f"Validation ROC-AUC: {roc_auc:.4f}")
        print(f"Chosen threshold: {threshold:.6f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(
            "True HIT games detected:",
            metrics["true_positives"],
        )
        print(
            "False positives:",
            metrics["false_positives"],
        )

        comparison_rows.append({
            "model": name,
            "validation_pr_auc": pr_auc,
            "validation_roc_auc": roc_auc,
            "threshold": threshold,
            "threshold_selection_reason": reason,
            **metrics,
        })

        # Save complete threshold curve for each candidate.
        threshold_table.insert(
            0,
            "model",
            name,
        )

        threshold_rows.append(threshold_table)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "validation_pr_auc",
        ascending=False,
    )

    comparison = comparison.reset_index(
        drop=True
    )

    comparison.insert(
        0,
        "rank",
        comparison.index + 1,
    )

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    all_thresholds = pd.concat(
        threshold_rows,
        ignore_index=True,
    )

    all_thresholds.to_csv(
        THRESHOLD_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION COMPARISON")
    print("=" * 65)

    print(
        comparison[
            [
                "rank",
                "model",
                "validation_pr_auc",
                "validation_roc_auc",
                "threshold",
                "precision",
                "recall",
                "true_positives",
                "false_positives",
            ]
        ].to_string(index=False)
    )

    best = comparison.iloc[0]

    print("\nBest validation PR-AUC:")
    print(best["model"])
    print(f"{best['validation_pr_auc']:.4f}")

    print("\nSaved comparison:", COMPARISON_FILE)
    print("Saved thresholds:", THRESHOLD_FILE)

    print(
        "\nExperiment complete. The test set was not used "
        "to select models or thresholds."
    )


if __name__ == "__main__":
    main()