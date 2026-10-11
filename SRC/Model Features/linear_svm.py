from pathlib import Path
import html
import inspect
import re

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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

FEATURE_FILE = Path(
    "Data/Processed/steam_features_pricefixed.csv"
)

TEXT_SOURCE_FILE = Path(
    "Data/Processed/steam_modeling_final_prices_fixed.csv"
)

COMPARISON_FILE = Path(
    "reports/linear_svm_comparison.csv"
)

THRESHOLD_FILE = Path(
    "reports/linear_svm_thresholds.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

SVM_C_VALUES = [0.03, 0.10, 0.30, 1.00]

MIN_PRECISION = 0.35

EXPECTED_TRAIN_ROWS = 13369
EXPECTED_VALIDATION_ROWS = 4372


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
# TEXT HELPERS
# ============================================================

def clean_text(value):
    """Remove HTML tags and normalize whitespace."""

    if not isinstance(value, str):
        return ""

    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def split_labels(value):
    """Split pipe-separated genre and category labels."""

    if not isinstance(value, str) or not value.strip():
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


# ============================================================
# STRUCTURED FEATURE PIPELINE
# ============================================================

def make_one_hot_encoder():
    """Create an encoder compatible with installed sklearn."""

    parameters = inspect.signature(
        OneHotEncoder
    ).parameters

    kwargs = {
        "handle_unknown": "ignore",
    }

    if "min_frequency" in parameters:
        kwargs["min_frequency"] = 5

    return OneHotEncoder(**kwargs)


def make_preprocessor():
    """Build transformations for non-description features."""

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


def prepare_features(df):
    """Select allowed features and normalize their types."""

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
            f"Missing required features: {missing}"
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
# THRESHOLD SELECTION
# ============================================================

def choose_threshold(y_true, scores):
    """
    Maximize recall while retaining at least 35% precision
    on validation data.

    LinearSVC produces decision scores rather than calibrated
    probabilities, so its threshold is on the score scale.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No threshold candidates were generated."
        )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
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

    if eligible.empty:
        selected = table.sort_values(
            ["f1", "precision"],
            ascending=[False, False],
        ).iloc[0]

        reason = "maximum_f1_fallback"

    else:
        selected = eligible.sort_values(
            ["recall", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

        reason = "maximum_recall_at_precision_floor"

    return (
        float(selected["threshold"]),
        reason,
        table,
    )


def calculate_threshold_metrics(
    y_true,
    scores,
    threshold,
):
    """Calculate classification metrics at the selected threshold."""

    actual = np.asarray(y_true)

    predicted = (
        np.asarray(scores) >= threshold
    ).astype(int)

    tp = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    fp = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    tn = int(
        ((actual == 0) & (predicted == 0)).sum()
    )

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
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("STEAMPULSE LINEAR SVM EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD STRUCTURED FEATURES
    # --------------------------------------------------------

    print("\nLoading corrected structured features...")

    df = pd.read_csv(
        FEATURE_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = [
        "steam_appid",
        "hit",
        "split",
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

    if df["steam_appid"].isna().any():
        raise ValueError(
            "Missing Steam app IDs."
        )

    if df["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate Steam app IDs."
        )

    df["steam_appid"] = pd.to_numeric(
        df["steam_appid"],
        errors="raise",
    ).astype("int64")

    df["hit"] = pd.to_numeric(
        df["hit"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # LOAD TITLES AND DESCRIPTIONS
    # --------------------------------------------------------

    print("Loading text data...")

    text_source = pd.read_csv(
        TEXT_SOURCE_FILE,
        encoding="utf-8-sig",
        usecols=[
            "steam_appid",
            "name",
            "short_description",
            "about_the_game",
        ],
        low_memory=False,
    )

    if text_source["steam_appid"].isna().any():
        raise ValueError(
            "Missing IDs in text source."
        )

    if text_source["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in text source."
        )

    text_source["steam_appid"] = pd.to_numeric(
        text_source["steam_appid"],
        errors="raise",
    ).astype("int64")

    for column in [
        "name",
        "short_description",
        "about_the_game",
    ]:
        text_source[column] = (
            text_source[column]
            .map(clean_text)
        )

    text_source["description_text"] = (
        text_source["short_description"]
        + " "
        + text_source["about_the_game"]
    ).str.strip()

    df = df.merge(
        text_source[
            [
                "steam_appid",
                "name",
                "description_text",
            ]
        ].rename(
            columns={"name": "original_title"}
        ),
        on="steam_appid",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # SPLIT DATA
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

    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} training records; "
            f"found {len(train):,}."
        )

    if len(validation) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_ROWS:,} validation records; "
            f"found {len(validation):,}."
        )

    y_train = train["hit"].astype(int)
    y_validation = validation["hit"].astype(int)

    print(f"Training games: {len(train):,}")
    print(f"Validation games: {len(validation):,}")
    print(f"Linear SVM C values: {SVM_C_VALUES}")

    # --------------------------------------------------------
    # PREPARE STRUCTURED FEATURES ONCE
    # --------------------------------------------------------

    print("\nPreparing structured feature matrix...")

    X_train = prepare_features(train)
    X_validation = prepare_features(validation)

    preprocessor = make_preprocessor()

    train_base = preprocessor.fit_transform(
        X_train,
        y_train,
    )

    validation_base = preprocessor.transform(
        X_validation
    )

    # --------------------------------------------------------
    # DESCRIPTION WORDS: TF-IDF
    # --------------------------------------------------------

    print("Fitting description-word TF-IDF once...")

    description_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.98,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    train_descriptions = (
        train["description_text"]
        .fillna("")
        .astype(str)
    )

    validation_descriptions = (
        validation["description_text"]
        .fillna("")
        .astype(str)
    )

    train_description_matrix = (
        description_vectorizer.fit_transform(
            train_descriptions
        )
    )

    validation_description_matrix = (
        description_vectorizer.transform(
            validation_descriptions
        )
    )

    # --------------------------------------------------------
    # TITLE CHARACTER N-GRAMS
    # --------------------------------------------------------

    print("Fitting title character n-grams once...")

    title_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=15000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    train_titles = (
        train["original_title"]
        .fillna("")
        .astype(str)
    )

    validation_titles = (
        validation["original_title"]
        .fillna("")
        .astype(str)
    )

    train_title_matrix = (
        title_vectorizer.fit_transform(
            train_titles
        )
    )

    validation_title_matrix = (
        title_vectorizer.transform(
            validation_titles
        )
    )

    # Keep all feature matrices sparse for efficiency.
    train_matrix = hstack(
        [
            train_base,
            train_description_matrix,
            train_title_matrix,
        ],
        format="csr",
    )

    validation_matrix = hstack(
        [
            validation_base,
            validation_description_matrix,
            validation_title_matrix,
        ],
        format="csr",
    )

    print("Combined training matrix:", train_matrix.shape)
    print("Combined validation matrix:", validation_matrix.shape)

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    comparison_rows = []
    threshold_tables = []

    # --------------------------------------------------------
    # BASELINE: LOGISTIC REGRESSION C=0.3
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("BASELINE: Logistic Regression C=0.3")
    print("-" * 65)

    baseline = LogisticRegression(
        C=0.3,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    baseline.fit(
        train_matrix,
        y_train,
    )

    baseline_scores = baseline.predict_proba(
        validation_matrix
    )[:, 1]

    baseline_pr_auc = average_precision_score(
        y_validation,
        baseline_scores,
    )

    baseline_roc_auc = roc_auc_score(
        y_validation,
        baseline_scores,
    )

    threshold, reason, threshold_table = (
        choose_threshold(
            y_validation,
            baseline_scores,
        )
    )

    metrics = calculate_threshold_metrics(
        y_validation,
        baseline_scores,
        threshold,
    )

    comparison_rows.append({
        "model": "Logistic Regression C=0.3",
        "C": 0.3,
        "validation_pr_auc": baseline_pr_auc,
        "validation_roc_auc": baseline_roc_auc,
        "threshold": threshold,
        "threshold_selection_reason": reason,
        **metrics,
    })

    threshold_table.insert(
        0,
        "model",
        "Logistic Regression C=0.3",
    )

    threshold_tables.append(threshold_table)

    print(f"Validation PR-AUC: {baseline_pr_auc:.4f}")
    print(f"Validation ROC-AUC: {baseline_roc_auc:.4f}")

    # --------------------------------------------------------
    # LINEAR SVM EXPERIMENTS
    # --------------------------------------------------------

    for index, c_value in enumerate(
        SVM_C_VALUES,
        start=1,
    ):

        print("\n" + "-" * 65)
        print(
            f"LINEAR SVM {index}/{len(SVM_C_VALUES)} "
            f"| C={c_value}"
        )
        print("-" * 65)

        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual=True,
            max_iter=10000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            train_matrix,
            y_train,
        )

        # LinearSVC returns decision scores, not probabilities.
        scores = model.decision_function(
            validation_matrix
        )

        pr_auc = average_precision_score(
            y_validation,
            scores,
        )

        roc_auc = roc_auc_score(
            y_validation,
            scores,
        )

        threshold, reason, threshold_table = (
            choose_threshold(
                y_validation,
                scores,
            )
        )

        metrics = calculate_threshold_metrics(
            y_validation,
            scores,
            threshold,
        )

        print(f"Validation PR-AUC: {pr_auc:.4f}")
        print(f"Validation ROC-AUC: {roc_auc:.4f}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(
            "True positives:",
            metrics["true_positives"],
        )
        print(
            "False positives:",
            metrics["false_positives"],
        )

        comparison_rows.append({
            "model": f"Linear SVM C={c_value}",
            "C": c_value,
            "validation_pr_auc": pr_auc,
            "validation_roc_auc": roc_auc,
            "threshold": threshold,
            "threshold_selection_reason": reason,
            **metrics,
        })

        threshold_table.insert(
            0,
            "model",
            f"Linear SVM C={c_value}",
        )

        threshold_tables.append(threshold_table)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "validation_pr_auc",
        ascending=False,
    ).reset_index(drop=True)

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
        threshold_tables,
        ignore_index=True,
    )

    all_thresholds.to_csv(
        THRESHOLD_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("LINEAR SVM COMPARISON")
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
                "f1",
                "true_positives",
                "false_positives",
            ]
        ].to_string(index=False)
    )

    best = comparison.iloc[0]

    print("\nBest candidate:", best["model"])
    print(
        "Best validation PR-AUC:",
        round(float(best["validation_pr_auc"]), 4),
    )

    print("\nSaved comparison:", COMPARISON_FILE)
    print("Saved threshold analysis:", THRESHOLD_FILE)

    print(
        "\nExperiment completed. Test data was not used "
        "to select a model or threshold."
    )


if __name__ == "__main__":
    main()