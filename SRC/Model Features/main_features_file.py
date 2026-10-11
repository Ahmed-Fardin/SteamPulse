import json
import re
import html
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# FILES
# ==================================================

PRICE_FILE = Path(
    "Data/Processed/steam_modeling_final_prices.csv"
)

LABEL_FILE = Path(
    "Data/Processed/steam_ml_integrated.csv"
)

OUTPUT_FILE = Path(
    "Data/Processed/steam_features_final.csv"
)

AUDIT_FILE = Path(
    "Data/Processed/feature_engineering_audit.csv"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def parse_json(value):
    """Parse a JSON string without crashing on bad values."""

    if isinstance(value, (dict, list)):
        return value

    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_bool(value):
    """Convert common representations of booleans."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def clean_text(value):
    """Remove HTML tags and normalize whitespace."""

    if not isinstance(value, str):
        return ""

    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def parse_name_list(value):
    """
    Extract names from Steam JSON lists.

    Supports lists containing strings or dictionaries
    with description/name fields.
    """

    parsed = parse_json(value)

    if parsed is None:
        return []

    if not isinstance(parsed, list):
        parsed = [parsed]

    names = []

    for item in parsed:
        if isinstance(item, dict):
            name = (
                item.get("description")
                or item.get("name")
                or item.get("title")
            )

        elif isinstance(item, str):
            name = item

        else:
            name = None

        if name:
            name = str(name).strip()

            if name and name not in names:
                names.append(name)

    return names


def parse_platforms(value):
    """Extract the supported operating systems."""

    parsed = parse_json(value)

    if not isinstance(parsed, dict):
        parsed = {}

    return {
        "windows": int(parse_bool(parsed.get("windows", False))),
        "mac": int(parse_bool(parsed.get("mac", False))),
        "linux": int(parse_bool(parsed.get("linux", False))),
    }


def has_requirements(value):
    """Determine whether nonempty system requirements exist."""

    parsed = parse_json(value)

    if isinstance(parsed, dict):
        return int(
            any(
                isinstance(v, str) and bool(clean_text(v))
                for v in parsed.values()
            )
        )

    if isinstance(parsed, list):
        return int(bool(parsed))

    if isinstance(value, str):
        return int(bool(clean_text(value)))

    return 0


def language_count(value):
    """Estimate the number of listed supported languages."""

    text = clean_text(value)

    if not text:
        return 0

    languages = [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]

    return len(set(languages))


def get_release_month(value):
    """Extract the release month when the date is parseable."""

    if not isinstance(value, str) or not value.strip():
        return np.nan

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        format="mixed"
    )

    if pd.isna(parsed):
        return np.nan

    return int(parsed.month)


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def engineer_features(df):
    """Build model-ready features while preserving game IDs."""

    print("Engineering features...")

    result = pd.DataFrame(index=df.index)

    # --------------------------------------------------
    # IDENTIFIERS AND TARGET METADATA
    # --------------------------------------------------

    result["steam_appid"] = df["steam_appid"]
    result["appid"] = df["appid"]

    result["name"] = (
        df["name"]
        .fillna("")
        .astype(str)
    )

    result["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    )

    # --------------------------------------------------
    # RELEASE MONTH
    # --------------------------------------------------

    result["release_month"] = (
        df["release_date"].apply(get_release_month)
        if "release_date" in df.columns
        else np.nan
    )

    # --------------------------------------------------
    # PRICE FEATURES
    # --------------------------------------------------

    result["price_usd_final"] = pd.to_numeric(
        df["price_usd_final"],
        errors="coerce"
    )

    result["price_log1p_final"] = pd.to_numeric(
        df["price_log1p_final"],
        errors="coerce"
    )

    result["price_missing_final"] = pd.to_numeric(
        df["price_missing_final"],
        errors="coerce"
    ).fillna(1).astype("int8")

    # Mark unusual prices but do not delete games.
    result["price_review_flag"] = (
        result["price_usd_final"].gt(500)
    ).astype("int8")

    # --------------------------------------------------
    # FREE-TO-PLAY STATUS
    # --------------------------------------------------

    if "is_free_clean" in df.columns:
        result["is_free"] = df["is_free_clean"].map(
            parse_bool
        ).astype("int8")

    else:
        result["is_free"] = df["is_free"].map(
            parse_bool
        ).astype("int8")

    # --------------------------------------------------
    # REQUIRED AGE
    # --------------------------------------------------

    result["required_age"] = pd.to_numeric(
        df["required_age"],
        errors="coerce"
    )

    # --------------------------------------------------
    # PLATFORMS
    # --------------------------------------------------

    platform_data = df["platforms"].apply(parse_platforms)

    for column in ["windows", "mac", "linux"]:
        result[column] = platform_data.apply(
            lambda item: item[column]
        )

    result["platform_count"] = (
        result["windows"]
        + result["mac"]
        + result["linux"]
    )

    # --------------------------------------------------
    # GENRES AND CATEGORIES
    # --------------------------------------------------

    genres = df["genres"].apply(parse_name_list)
    categories = df["categories"].apply(parse_name_list)

    result["genres_multilabel"] = genres.apply(
        lambda values: "|".join(values)
    )

    result["categories_multilabel"] = categories.apply(
        lambda values: "|".join(values)
    )

    result["genre_count"] = genres.apply(len)
    result["category_count"] = categories.apply(len)

    result["main_genre"] = genres.apply(
        lambda values: values[0] if values else "Unknown"
    )

    result["main_category"] = categories.apply(
        lambda values: values[0] if values else "Unknown"
    )

    # --------------------------------------------------
    # DEVELOPERS AND PUBLISHERS
    # --------------------------------------------------

    developers = df["developers"].apply(parse_name_list)
    publishers = df["publishers"].apply(parse_name_list)

    result["developer_count"] = developers.apply(len)
    result["publisher_count"] = publishers.apply(len)

    result["main_developer"] = developers.apply(
        lambda values: values[0] if values else "Unknown"
    )

    result["main_publisher"] = publishers.apply(
        lambda values: values[0] if values else "Unknown"
    )

    # --------------------------------------------------
    # LANGUAGES
    # --------------------------------------------------

    result["language_count"] = df[
        "supported_languages"
    ].apply(language_count)

    # --------------------------------------------------
    # CONTROLLER SUPPORT
    # --------------------------------------------------

    result["has_controller_support"] = (
        df["controller_support"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype("int8")
    )

    # --------------------------------------------------
    # SYSTEM REQUIREMENTS
    # --------------------------------------------------

    result["has_pc_requirements"] = df[
        "pc_requirements"
    ].apply(has_requirements)

    result["has_mac_requirements"] = df[
        "mac_requirements"
    ].apply(has_requirements)

    result["has_linux_requirements"] = df[
        "linux_requirements"
    ].apply(has_requirements)

    # --------------------------------------------------
    # TEXT-LENGTH FEATURES
    # --------------------------------------------------

    result["name_length"] = result["name"].apply(
        lambda value: len(clean_text(value))
    )

    result["short_description_length"] = df[
        "short_description"
    ].apply(
        lambda value: len(clean_text(value))
    )

    result["about_length"] = df[
        "about_the_game"
    ].apply(
        lambda value: len(clean_text(value))
    )

    return result


# ==================================================
# MAIN WORKFLOW
# ==================================================

def main():

    print("Loading full modeling population...")

    games = pd.read_csv(
        PRICE_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    print("Loading labeled outcomes...")

    labels = pd.read_csv(
        LABEL_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    # --------------------------------------------------
    # CHECK APP IDS
    # --------------------------------------------------

    key = "steam_appid"

    for name, data in [
        ("games", games),
        ("labels", labels),
    ]:

        if key not in data.columns:
            raise ValueError(
                f"{name} is missing {key}"
            )

        data[key] = pd.to_numeric(
            data[key],
            errors="raise"
        ).astype("int64")

        if data[key].isna().any():
            raise ValueError(
                f"{name} contains missing game IDs"
            )

        if data[key].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate game IDs"
            )

    # --------------------------------------------------
    # RETAIN THE EXISTING LABELS
    # --------------------------------------------------

    label_columns = [
        "steam_appid",
        "hit",
        "split"
    ]

    label_data = labels[label_columns].copy()

    label_data["hit"] = pd.to_numeric(
        label_data["hit"],
        errors="raise"
    )

    # Avoid overwriting the original split.
    label_data = label_data.rename(
        columns={"split": "labeled_split"}
    )

    games = games.merge(
        label_data,
        on="steam_appid",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # VALIDATE SPLITS
    # --------------------------------------------------

    mismatch = (
        games["labeled_split"].notna()
        & games["split"].notna()
        & games["labeled_split"].ne(games["split"])
    )

    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} labeled games have "
            "inconsistent temporal split assignments."
        )

    games = games.drop(columns=["labeled_split"])

    # --------------------------------------------------
    # ENGINEER FEATURES
    # --------------------------------------------------

    features = engineer_features(games)

    # Attach target and split metadata.
    features["split"] = games["split"].values
    features["hit"] = games["hit"].values

    # Only 1 or 0 is a valid observed label.
    valid_labels = features["hit"].isin([0, 1])

    features["has_observed_outcome"] = (
        valid_labels
    ).astype("int8")

    # Keep unlabeled outcomes genuinely missing.
    features.loc[
        ~valid_labels,
        "hit"
    ] = np.nan

    # --------------------------------------------------
    # BASIC INTEGRITY CHECKS
    # --------------------------------------------------

    if len(features) != len(games):
        raise RuntimeError(
            "Feature engineering changed the row count."
        )

    if features["steam_appid"].duplicated().any():
        raise RuntimeError(
            "Duplicate game IDs appeared in final features."
        )

    labeled_count = int(valid_labels.sum())
    unlabeled_count = int((~valid_labels).sum())

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Report feature availability and split distribution.
    audit_rows = []

    audit_rows.append({
        "check": "total_games",
        "value": len(features)
    })

    audit_rows.append({
        "check": "labeled_games",
        "value": labeled_count
    })

    audit_rows.append({
        "check": "unlabeled_games",
        "value": unlabeled_count
    })

    audit_rows.append({
        "check": "duplicate_app_ids",
        "value": int(features["steam_appid"].duplicated().sum())
    })

    audit_rows.append({
        "check": "missing_price_features",
        "value": int(features["price_missing_final"].sum())
    })

    for split_name, count in (
        features["split"].value_counts(dropna=False).items()
    ):
        audit_rows.append({
            "check": f"split_{split_name}",
            "value": int(count)
        })

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\nFEATURE ENGINEERING COMPLETE")

    print(f"Total games: {len(features):,}")
    print(f"Labeled games: {labeled_count:,}")
    print(f"Unlabeled games: {unlabeled_count:,}")

    print("\nLabeled games by split:")

    print(
        features.loc[valid_labels, "split"]
        .value_counts()
        .to_string()
    )

    print("\nOutput columns:")
    print(features.columns.tolist())

    print(f"\nSaved features: {OUTPUT_FILE}")
    print(f"Saved audit: {AUDIT_FILE}")

    print(
        "\nNote: Recommendation counts, reviews, ratings, "
        "and Metacritic scores are not model features."
    )


if __name__ == "__main__":
    main()