from pathlib import Path
import inspect
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FILES
# ============================================================

FEATURE_FILE = Path(
    "Data/Processed/steam_features_pricefixed.csv"
)

COMPARISON_FILE = Path(
    "reports/price_ablation_comparison.csv"
)

MODEL_FILE = Path(
    "models/steampulse_selected_model.pkl"
)

METRICS_FILE = Path(
    "reports/selected_model_metrics.json"
)

TEST_PREDICTIONS_FILE = Path(
    "Data/Processed/steampulse_selected_test_predictions.csv"
)

POPULATION_PREDICTIONS_FILE = Path(
    "Data/Processed/steampulse_selected_predictions.csv"
)


# ============================================================
# FIXED CONFIGURATION
# ============================================================

RANDOM_STATE = 42

SELECTED_EXPERIMENT = (
    "Logistic Regression - corrected price"
)

EXPECTED_SPLIT_COUNTS = {
    "train": 13369,
    "validation": 4372,
    "test": 4884,
}


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
# DATA PREPARATION
# ============================================================

def prepare_features(df):
    """Prepare the permitted input columns."""

    columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + MULTILABEL_FEATURES
    )

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X = df[columns].copy()

    for column in NUMERIC_FEATURES:
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
# PREPROCESSOR AND MODEL
# ============================================================

def make_one_hot_encoder():
    """Support multiple scikit-learn versions."""

    parameters = inspect.signature(
        OneHotEncoder
    ).parameters

    kwargs = {
        "handle_unknown": "ignore",
    }

    if "min_frequency" in parameters:
        kwargs["min_frequency"] = 5

    return OneHotEncoder(**kwargs)


def build_model():
    """
    Build the selected Logistic Regression candidate.

    The multilabel token patterns treat each pipe-separated
    genre/category as one token, including multiword labels.
    """

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
            (
                "genres",
                CountVectorizer(
                    token_pattern=r"[^|]+",
                    lowercase=False,
                    binary=True,
                    min_df=5,
                ),
                "genres_multilabel",
            ),
            (
                "categories",
                CountVectorizer(
                    token_pattern=r"[^|]+",
                    lowercase=False,
                    binary=True,
                    min_df=5,
                ),
                "categories_multilabel",
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    """Calculate ranking and classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc_average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "hit_precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "hit_recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "hit_f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "predicted_hit_count": int(
            predictions.sum()
        ),
        "actual_hit_count": int(
            np.sum(y_true)
        ),
        "actual_hit_prevalence": float(
            np.mean(y_true)
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["NOT HIT", "HIT"],
            output_dict=True,
            zero_division=0,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STEAMPULSE SELECTED MODEL")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading corrected feature dataset...")

    df = pd.read_csv(
        FEATURE_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = [
        "steam_appid",
        "appid",
        "name",
        "release_year",
        "split",
        "hit",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["hit"] = pd.to_numeric(
        df["hit"],
        errors="coerce",
    )

    if df["steam_appid"].isna().any():
        raise ValueError(
            "Missing Steam app IDs."
        )

    if df["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate Steam app IDs."
        )

    # --------------------------------------------------------
    # LOAD SELECTED VALIDATION THRESHOLD
    # --------------------------------------------------------

    comparison = pd.read_csv(
        COMPARISON_FILE,
        encoding="utf-8-sig",
    )

    selected = comparison[
        comparison["model"] == SELECTED_EXPERIMENT
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one row for the selected experiment."
        )

    threshold = float(
        selected.iloc[0]["threshold"]
    )

    print(f"Selected model: {SELECTED_EXPERIMENT}")
    print(f"Validation PR-AUC: "
          f"{selected.iloc[0]['validation_pr_auc']:.4f}")
    print(f"Validation ROC-AUC: "
          f"{selected.iloc[0]['validation_roc_auc']:.4f}")
    print(f"Selected threshold: {threshold:.6f}")

    # --------------------------------------------------------
    # CREATE SPLITS
    # --------------------------------------------------------

    labeled = df[
        df["hit"].isin([0, 1])
    ].copy()

    train = labeled[
        labeled["split"] == "train"
    ].copy()

    validation = labeled[
        labeled["split"] == "validation"
    ].copy()

    test = labeled[
        labeled["split"] == "test"
    ].copy()

    for split_name, expected in EXPECTED_SPLIT_COUNTS.items():
        actual = int(
            (labeled["split"] == split_name).sum()
        )

        if actual != expected:
            raise ValueError(
                f"{split_name} count changed: "
                f"expected {expected}, found {actual}."
            )

    print(f"\nTraining games: {len(train):,}")
    print(f"Validation games: {len(validation):,}")
    print(f"Test games: {len(test):,}")
    print(f"Total modeling population: {len(df):,}")

    # --------------------------------------------------------
    # FIT ON TRAINING + VALIDATION
    # --------------------------------------------------------
    # The candidate and threshold were selected using
    # training/validation only. The test set is not used
    # to fit this model.

    train_validation = pd.concat(
        [train, validation],
        axis=0,
    )

    X_train_validation = prepare_features(
        train_validation
    )

    y_train_validation = (
        train_validation["hit"].astype(int)
    )

    model = build_model()

    print("\nTraining selected model on "
          "training + validation data...")

    model.fit(
        X_train_validation,
        y_train_validation,
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating on the existing test set...")

    X_test = prepare_features(test)
    y_test = test["hit"].astype(int)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(
        f"PR-AUC: "
        f"{test_metrics['pr_auc_average_precision']:.4f}"
    )
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"HIT precision: "
        f"{test_metrics['hit_precision']:.4f}"
    )
    print(
        f"HIT recall: "
        f"{test_metrics['hit_recall']:.4f}"
    )
    print(f"HIT F1: {test_metrics['hit_f1']:.4f}")

    print("\nConfusion matrix:")
    print(
        np.array([
            [
                test_metrics["true_negative"],
                test_metrics["false_positive"],
            ],
            [
                test_metrics["false_negative"],
                test_metrics["true_positive"],
            ],
        ])
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    test_output = test[
        [
            "steam_appid",
            "appid",
            "name",
            "release_year",
            "split",
            "hit",
        ]
    ].copy()

    test_output["predicted_hit_probability"] = (
        test_probabilities
    )

    test_output["predicted_hit"] = (
        test_probabilities >= threshold
    ).astype(int)

    test_output["decision_threshold"] = threshold

    TEST_PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_output.to_csv(
        TEST_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # FULL MODELING-POPULATION PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating full modeling-population predictions...")

    X_all = prepare_features(df)

    all_probabilities = model.predict_proba(
        X_all
    )[:, 1]

    population_predictions = df[
        [
            "steam_appid",
            "appid",
            "name",
            "release_year",
            "split",
            "hit",
        ]
    ].copy()

    population_predictions[
        "predicted_hit_probability"
    ] = all_probabilities

    population_predictions["predicted_hit"] = (
        all_probabilities >= threshold
    ).astype(int)

    population_predictions[
        "decision_threshold"
    ] = threshold

    valid_label_mask = population_predictions[
        "hit"
    ].isin([0, 1])

    population_predictions["prediction_status"] = (
        np.where(
            ~valid_label_mask,
            "unlabeled_game_estimate",
            np.where(
                population_predictions["split"].eq("test"),
                "held_out_test_prediction",
                "labeled_training_population_prediction",
            ),
        )
    )

    POPULATION_PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    population_predictions.to_csv(
        POPULATION_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # SAVE MODEL BUNDLE
    # --------------------------------------------------------

    model_bundle = {
        "model": model,
        "model_name": "Logistic Regression",
        "feature_mode": "all",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "multilabel_features": MULTILABEL_FEATURES,
        "threshold": threshold,
        "target_definition": (
            "HIT = observed recommendation_count >= 6785"
        ),
        "split_definition": {
            "train": "2000-2022",
            "validation": "2023-2024",
            "test": "2025-2026",
        },
        "validation_pr_auc": float(
            selected.iloc[0]["validation_pr_auc"]
        ),
        "validation_roc_auc": float(
            selected.iloc[0]["validation_roc_auc"]
        ),
        "notes": [
            (
                "Model was refitted on training plus validation."
            ),
            (
                "Decision threshold was selected on validation "
                "predictions before this refit."
            ),
            (
                "Current prices and metadata are not verified "
                "historical snapshots from each game's release."
            ),
            (
                "HIT represents cumulative recommendation "
                "threshold attainment, not commercial success."
            ),
            (
                "The existing test set has been evaluated by "
                "earlier models during development."
            ),
        ],
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_bundle,
        MODEL_FILE,
        compress=3,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_report = {
        "selected_model": "Logistic Regression",
        "validation_pr_auc": float(
            selected.iloc[0]["validation_pr_auc"]
        ),
        "validation_roc_auc": float(
            selected.iloc[0]["validation_roc_auc"]
        ),
        "threshold_selected_on_validation": threshold,
        "training_games": len(train),
        "validation_games": len(validation),
        "test_games": len(test),
        "population_games": len(df),
        "test_metrics": test_metrics,
        "caveats": model_bundle["notes"],
    }

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_report,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("FINAL MODEL SAVED")
    print("=" * 65)

    print(f"Model: {MODEL_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Test predictions: {TEST_PREDICTIONS_FILE}")
    print(
        f"Population predictions: "
        f"{POPULATION_PREDICTIONS_FILE}"
    )

    print("\nAll source datasets remain unchanged.")


if __name__ == "__main__":
    main()