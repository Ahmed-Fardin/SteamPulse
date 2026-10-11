import json
import pandas as pd


input_file = "Data/Processed/steam_ml.csv"
output_file = "Data/Processed/steam_features.csv"


df = pd.read_csv(
    input_file,
    encoding="utf-8",
    encoding_errors="replace"
)

print("Building SteamPulse features...")


# Age
df["required_age"] = (
    df["required_age"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)


# Free game
df["is_free"] = (
    df["is_free"]
    .astype(str)
    .str.lower()
    .map({"true": 1, "false": 0})
)


# Price
def get_price(value):
    try:
        data = json.loads(value)

        if isinstance(data, dict):
            return (data.get("initial") or 0) / 100

    except:
        pass

    return 0


df["price"] = df["price_overview"].apply(get_price)


# Platforms
def get_platforms(value):
    try:
        return json.loads(value)
    except:
        return {}


platforms = df["platforms"].apply(get_platforms)

df["windows"] = platforms.apply(lambda x: int(x.get("windows", False)))
df["mac"] = platforms.apply(lambda x: int(x.get("mac", False)))
df["linux"] = platforms.apply(lambda x: int(x.get("linux", False)))

df["platform_count"] = (
    df["windows"] +
    df["mac"] +
    df["linux"]
)


# Developers and publishers
def count_items(value):
    try:
        return len(json.loads(value))
    except:
        return 0


df["developer_count"] = df["developers"].apply(count_items)
df["publisher_count"] = df["publishers"].apply(count_items)


# Genres and categories
df["genre_count"] = df["genres"].apply(count_items)
df["category_count"] = df["categories"].apply(count_items)


# Languages
df["language_count"] = (
    df["supported_languages"]
    .fillna("")
    .apply(
        lambda x: len(
            [i for i in x.split(",") if i.strip()]
        )
    )
)


# Controller support
df["has_controller_support"] = (
    df["controller_support"]
    .fillna("")
    .str.strip()
    .ne("")
    .astype(int)
)


# System requirements
df["has_pc_requirements"] = (
    df["pc_requirements"]
    .fillna("")
    .str.strip()
    .ne("")
    .astype(int)
)

df["has_mac_requirements"] = (
    df["mac_requirements"]
    .fillna("")
    .str.strip()
    .ne("")
    .astype(int)
)

df["has_linux_requirements"] = (
    df["linux_requirements"]
    .fillna("")
    .str.strip()
    .ne("")
    .astype(int)
)


# Description features
df["name_length"] = df["name"].fillna("").str.len()

df["short_description_length"] = (
    df["short_description"]
    .fillna("")
    .str.len()
)

df["about_length"] = (
    df["about_the_game"]
    .fillna("")
    .str.len()
)


# Save
df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("Done.")
print(f"Rows: {len(df):,}")
print(f"Saved to: {output_file}")