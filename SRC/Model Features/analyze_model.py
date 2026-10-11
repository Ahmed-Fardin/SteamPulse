from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# FILES
# ============================================================

FEATURE_FILE = Path(
    "Data/Processed/steam_features_final.csv"
)

MODEL_FILE = Path(
    "models/steampulse_tuned_model.pkl"
)

OUTPUT_FILE = Path(
    "reports/validation_permutation_importance.csv"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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
# TOKENIZER
# ============================================================

def split_labels(value):
    """Split pipe-delimited genre/category strings."""

    if not isinstance(value, str) or not value.strip():
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + MULTILABEL_FEATURES
    )

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X = df[feature_columns].copy()

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
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STEAMPULSE — VALIDATION PERMUTATION IMPORTANCE")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA AND SAVED MODEL CONFIGURATION
    # --------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        FEATURE_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print("Loading tuned model configuration...")

    bundle = joblib.load(MODEL_FILE)

    saved_model = bundle["model"]
    model_name = bundle["model_name"]
    feature_mode = bundle["feature_mode"]

    print(f"Model configuration: {model_name}")
    print(f"Feature mode: {feature_mode}")

    if feature_mode != "all":
        raise ValueError(
            "This script expects a model using all feature groups."
        )

    # --------------------------------------------------------
    # SELECT TRAINING AND VALIDATION RECORDS
    # --------------------------------------------------------

    df["hit"] = pd.to_numeric(
        df["hit"],
        errors="coerce",
    )

    train = df[
        (df["split"] == "train")
        & df["hit"].isin([0, 1])
    ].copy()

    validation = df[
        (df["split"] == "validation")
        & df["hit"].isin([0, 1])
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "Training or validation data is empty."
        )

    if train["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate training game IDs detected."
        )

    if validation["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate validation game IDs detected."
        )

    print(f"\nTraining games: {len(train):,}")
    print(f"Validation games: {len(validation):,}")

    X_train = prepare_features(train)
    X_validation = prepare_features(validation)

    y_train = train["hit"].astype(int)
    y_validation = validation["hit"].astype(int)

    # --------------------------------------------------------
    # REFIT ON TRAINING DATA ONLY
    # --------------------------------------------------------
    # The saved model was fitted on training + validation.
    # Cloning resets the fitted state but preserves its
    # selected hyperparameters and preprocessing configuration.

    print("\nFitting the selected model on training data only...")

    model = clone(saved_model)

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDATION BASELINE
    # --------------------------------------------------------

    print("\nCalculating baseline validation performance...")

    baseline_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    baseline_pr_auc = average_precision_score(
        y_validation,
        baseline_probabilities,
    )

    baseline_roc_auc = roc_auc_score(
        y_validation,
        baseline_probabilities,
    )

    print(f"Baseline validation PR-AUC: {baseline_pr_auc:.4f}")
    print(f"Baseline validation ROC-AUC: {baseline_roc_auc:.4f}")

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------
    # A positive importance means shuffling that feature
    # reduced the selected evaluation score.
    #
    # Negative importance means shuffling it improved the
    # score on average, which can indicate noise or instability.

    print("\nCalculating permutation importance...")
    print("Scoring metric: average precision")
    print("Permutations per feature: 7")

    results = permutation_importance(
        estimator=model,
        X=X_validation,
        y=y_validation,
        scoring="average_precision",
        n_repeats=7,
        random_state=42,
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "feature": X_validation.columns,
        "importance_mean": results.importances_mean,
        "importance_std": results.importances_std,
    })

    importance_df["baseline_validation_pr_auc"] = (
        baseline_pr_auc
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["rank"] = (
        importance_df.index + 1
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TOP 20 VALIDATION FEATURES")
    print("=" * 65)

    print(
        importance_df[
            [
                "rank",
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved report:")
    print(OUTPUT_FILE)

    print(
        "\nInterpretation: importance estimates how much "
        "validation average precision changes when a feature "
        "is shuffled. Correlated features can share predictive "
        "information, so a small individual score does not "
        "necessarily mean that a feature is useless."
    )


if __name__ == "__main__":
    main()