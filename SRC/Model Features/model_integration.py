from pathlib import Path

import pandas as pd


# --------------------------------------------------
# FILES
# --------------------------------------------------

LABEL_FILE = Path("Data/Processed/steam_ml.csv")

PRICE_FILE = Path(
    "Data/Processed/steam_modeling_final_prices.csv"
)

OUTPUT_FILE = Path(
    "Data/Processed/steam_ml_integrated.csv"
)

AUDIT_FILE = Path(
    "Data/Processed/integration_audit.csv"
)


def main():
    print("Loading labeled dataset...")

    labels = pd.read_csv(
        LABEL_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    print("Loading pricing dataset...")

    prices = pd.read_csv(
        PRICE_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    # --------------------------------------------------
    # VALIDATE KEYS
    # --------------------------------------------------

    key = "steam_appid"

    for name, df in [
        ("labels", labels),
        ("prices", prices)
    ]:
        if key not in df.columns:
            raise ValueError(
                f"{name} does not contain {key}"
            )

        if df[key].isna().any():
            raise ValueError(
                f"{name} contains missing Steam app IDs"
            )

        if df[key].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate Steam app IDs"
            )

    # IDs should have the same type before joining.
    labels[key] = pd.to_numeric(
        labels[key],
        errors="raise"
    ).astype("int64")

    prices[key] = pd.to_numeric(
        prices[key],
        errors="raise"
    ).astype("int64")

    # --------------------------------------------------
    # CHECK MATCHING IDS
    # --------------------------------------------------

    label_ids = set(labels[key])
    price_ids = set(prices[key])

    matched_ids = label_ids & price_ids
    unmatched_ids = label_ids - price_ids

    print("\nID MATCH AUDIT")
    print(f"Labeled games: {len(labels):,}")
    print(f"Price records: {len(prices):,}")
    print(f"Matched labeled games: {len(matched_ids):,}")
    print(f"Unmatched labeled games: {len(unmatched_ids):,}")

    # --------------------------------------------------
    # SELECT PRICE FEATURES
    # --------------------------------------------------

    price_columns = [
        "steam_appid",
        "price_usd_final",
        "price_log1p_final",
        "price_missing_final",
        "final_price_status",
        "is_free_clean"
    ]

    available_columns = [
        column
        for column in price_columns
        if column in prices.columns
    ]

    price_features = prices[available_columns].copy()

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    print("\nMerging datasets...")

    merged = labels.merge(
        price_features,
        on=key,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    merge_counts = (
        merged["_merge"]
        .value_counts()
        .rename_axis("merge_status")
        .reset_index(name="games")
    )

    merged["price_data_matched"] = (
        merged["_merge"] == "both"
    ).astype("int8")

    merged.drop(columns=["_merge"], inplace=True)

    # --------------------------------------------------
    # VALIDATE TARGET AND SPLIT
    # --------------------------------------------------

    if len(merged) != len(labels):
        raise RuntimeError(
            "Merge changed the labeled row count."
        )

    if merged["steam_appid"].duplicated().any():
        raise RuntimeError(
            "Duplicate IDs appeared after merging."
        )

    if not merged["hit"].equals(labels["hit"]):
        raise RuntimeError(
            "Target labels changed during integration."
        )

    if not merged["split"].equals(labels["split"]):
        raise RuntimeError(
            "Temporal split changed during integration."
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    merge_counts.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nINTEGRATION COMPLETE")
    print(f"Original labeled rows: {len(labels):,}")
    print(f"Integrated rows: {len(merged):,}")
    print(
        "Rows with price records:",
        int(merged["price_data_matched"].sum())
    )
    print(
        "Rows without price records:",
        int((merged["price_data_matched"] == 0).sum())
    )

    print("\nMerge audit:")
    print(merge_counts.to_string(index=False))

    print("\nSplit counts:")
    print(merged["split"].value_counts().to_string())

    print(f"\nSaved integrated data: {OUTPUT_FILE}")
    print(f"Saved audit: {AUDIT_FILE}")


if __name__ == "__main__":
    main()