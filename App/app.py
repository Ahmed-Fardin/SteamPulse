"""
SteamPulse — Steam Game Intelligence
=====================================

An interactive Streamlit application for exploring Steam games,
catalogue-level model predictions, and model evaluation.

Pages:
    1. Overview
    2. Explore Catalogue
    3. Game Spotlight
    4. Model Performance

Launch from the SteamPulse project root:

    F:\\Anaconda\\python.exe -m streamlit run App/app.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

ROOT = (
    APP_DIR.parent
    if APP_DIR.name.lower() == "app"
    else APP_DIR
)

PREDICTIONS_PATH = (
    ROOT / "Data" / "Processed"
    / "steampulse_catalog_predictions.csv"
)

MODEL_METRICS_PATH = (
    ROOT / "reports"
    / "steampulse_release_candidate_validation.json"
)

SCORING_SUMMARY_PATH = (
    ROOT / "reports"
    / "steampulse_catalog_scoring_summary.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SteamPulse | Game Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
        );

        html, body, [class*="css"] {
            font-family: Inter, sans-serif;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.20);
        }

        .hero {
            padding: 2.1rem;
            border-radius: 20px;
            background: linear-gradient(
                125deg,
                #111827 0%,
                #1e293b 55%,
                #312e81 100%
            );
            color: #f8fafc;
            margin-bottom: 1.8rem;
        }

        .eyebrow {
            color: #a5b4fc;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: clamp(2.2rem, 5vw, 3.7rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            line-height: 1.1;
        }

        .hero-text {
            max-width: 780px;
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.8;
            margin-top: 1rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin-bottom: 0.3rem;
        }

        .section-subtitle {
            opacity: 0.75;
            line-height: 1.7;
            margin-bottom: 1.2rem;
        }

        .footnote {
            font-size: 0.82rem;
            opacity: 0.72;
            line-height: 1.7;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 13px;
            padding: 1rem;
        }

        .footer {
            text-align: center;
            font-size: 0.78rem;
            opacity: 0.65;
            padding: 1.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading SteamPulse catalogue...")
def load_predictions(file_path, modified_time):
    """Load and normalize the precomputed prediction dataset."""

    del modified_time

    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    required = [
        "steam_appid",
        "name",
        "model_rank",
        "decision_score",
        "predicted_hit",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {missing}"
        )

    if df["steam_appid"].duplicated().any():
        raise ValueError(
            "Duplicate Steam App IDs found in the prediction data."
        )

    numeric_columns = [
        "steam_appid",
        "model_rank",
        "release_year",
        "release_month",
        "decision_score",
        "predicted_hit",
        "observed_hit",
        "platform_count",
        "genre_count",
        "category_count",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["steam_appid"] = df["steam_appid"].astype("int64")

    df["name"] = (
        df["name"]
        .fillna("Unknown title")
        .astype(str)
    )

    df["predicted_hit"] = (
        df["predicted_hit"]
        .fillna(0)
        .astype(int)
    )

    category_columns = [
        "main_genre",
        "main_category",
        "main_developer",
        "main_publisher",
    ]

    for column in category_columns:

        if column not in df.columns:
            df[column] = "Unknown"

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .replace(
                {
                    "": "Unknown",
                    "nan": "Unknown",
                    "None": "Unknown",
                }
            )
        )

    def normalize_free_status(value):

        if pd.isna(value):
            return "Unknown"

        if isinstance(value, (bool, np.bool_)):
            return "Free-to-play" if value else "Paid"

        value = str(value).strip().lower()

        if value in {"true", "1", "yes", "free"}:
            return "Free-to-play"

        if value in {"false", "0", "no", "paid"}:
            return "Paid"

        return "Unknown"

    if "is_free" in df.columns:
        df["free_status"] = df["is_free"].apply(
            normalize_free_status
        )
    else:
        df["free_status"] = "Unknown"

    if "observed_hit" not in df.columns:
        df["observed_hit"] = np.nan

    df["observed_hit"] = pd.to_numeric(
        df["observed_hit"],
        errors="coerce",
    )

    df["observed_hit"] = (
        df["observed_hit"]
        .where(df["observed_hit"].isin([0, 1]))
    )

    df["has_observed_outcome"] = (
        df["observed_hit"].notna()
    )

    df["observed_outcome_status"] = (
        df["observed_hit"]
        .map(
            {
                1: "Observed HIT",
                0: "Observed NOT HIT",
            }
        )
        .fillna("Not observed")
    )

    df["prediction_label"] = np.where(
        df["predicted_hit"] == 1,
        "Predicted HIT",
        "Not flagged as HIT",
    )

    return df


@st.cache_data(show_spinner=False)
def load_json(file_path, modified_time):
    """Load an optional JSON summary file."""

    del modified_time

    path = Path(file_path)

    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError):
        return {}


# ============================================================
# INITIALIZE DATA
# ============================================================

if not PREDICTIONS_PATH.is_file():

    st.error(
        "SteamPulse could not find its prediction dataset."
    )

    st.code(str(PREDICTIONS_PATH))

    st.info(
        "Check that catalog scoring completed successfully "
        "and the prediction CSV is in Data/Processed."
    )

    st.stop()


try:

    predictions = load_predictions(
        str(PREDICTIONS_PATH),
        PREDICTIONS_PATH.stat().st_mtime,
    )

    model_metrics = load_json(
        str(MODEL_METRICS_PATH),
        (
            MODEL_METRICS_PATH.stat().st_mtime
            if MODEL_METRICS_PATH.exists()
            else 0
        ),
    )

    scoring_summary = load_json(
        str(SCORING_SUMMARY_PATH),
        (
            SCORING_SUMMARY_PATH.stat().st_mtime
            if SCORING_SUMMARY_PATH.exists()
            else 0
        ),
    )

except Exception as error:

    st.error("SteamPulse could not load its application data.")

    st.exception(error)

    st.stop()


# These values are initialized only after the CSV has loaded.
total_games = len(predictions)

predicted_hit_count = int(
    predictions["predicted_hit"].sum()
)

predicted_hit_fraction = (
    predicted_hit_count / total_games
    if total_games
    else 0.0
)

observed_count = int(
    predictions["has_observed_outcome"].sum()
)

unobserved_count = total_games - observed_count

available_years = (
    predictions["release_year"]
    .dropna()
    .astype(int)
    .unique()
)

if len(available_years):

    minimum_year = int(np.min(available_years))
    maximum_year = int(np.max(available_years))

else:

    minimum_year = 2000
    maximum_year = 2026


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Use st.html for this block so HTML isn't parsed as Markdown.
    st.html(
        """
        <div style="padding: 0.6rem 0 1rem 0;">
            <div style="
                font-size: 1.7rem;
                font-weight: 800;
                letter-spacing: -0.06em;
            ">🎮 SteamPulse</div>
            <div style="
                opacity: 0.65;
                font-size: 0.8rem;
                margin-top: 0.25rem;
            ">Steam Game Intelligence</div>
        </div>
        """
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Explore Catalogue",
            "Game Spotlight",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("CATALOGUE")

    st.metric(
        "Games scored",
        f"{total_games:,}",
    )

    st.metric(
        "Predicted HIT",
        f"{predicted_hit_count:,}",
    )

    st.caption(
        "Predictions are precomputed. Opening the app "
        "does not retrain the model."
    )


# ============================================================
# COMMON HERO
# ============================================================

# Use Streamlit's HTML renderer to prevent Markdown from
# displaying nested HTML tags as literal text.
st.html(
    """
    <div class="hero">
        <div class="eyebrow">DATA SCIENCE · MACHINE LEARNING · GAMING</div>
        <div class="hero-title">SteamPulse</div>
        <div class="hero-text">
            Explore Steam games, uncover patterns in game metadata,
            and investigate the titles our model flags as potential
            high-engagement candidates.
        </div>
    </div>
    """
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="eyebrow">THE BIG PICTURE</div>
        <div class="section-title">Catalogue overview</div>
        <div class="section-subtitle">
            Explore the catalogue and see how the model classifies games.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Games scored", f"{total_games:,}")

    col2.metric(
        "Predicted HIT",
        f"{predicted_hit_count:,}",
        f"{predicted_hit_fraction:.2%} of scored games",
    )

    col3.metric(
        "Observed outcomes",
        f"{observed_count:,}",
    )

    col4.metric(
        "Unobserved outcomes",
        f"{unobserved_count:,}",
    )

    st.divider()

    left_chart, right_chart = st.columns(2)

    with left_chart:

        st.subheader("Model classification")

        class_data = pd.DataFrame(
            {
                "Classification": [
                    "Predicted HIT",
                    "Not flagged as HIT",
                ],
                "Games": [
                    predicted_hit_count,
                    total_games - predicted_hit_count,
                ],
            }
        )

        fig = px.pie(
            class_data,
            names="Classification",
            values="Games",
            hole=0.66,
            color="Classification",
            color_discrete_map={
                "Predicted HIT": "#818cf8",
                "Not flagged as HIT": "#334155",
            },
        )

        fig.update_traces(
            textinfo="percent",
            hovertemplate=(
                "%{label}<br>"
                "Games: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.caption(
            "Classifications use the saved decision threshold."
        )

    with right_chart:

        st.subheader("Predictions by release year")

        yearly_data = predictions.dropna(
            subset=["release_year"]
        ).copy()

        if not yearly_data.empty:

            yearly_data["release_year"] = (
                yearly_data["release_year"].astype(int)
            )

            yearly_stats = (
                yearly_data.groupby("release_year")
                .agg(
                    games=("steam_appid", "count"),
                    flagged_hit_rate=("predicted_hit", "mean"),
                )
                .reset_index()
            )

            # Hide small release-year groups from this chart only.
            # All games remain available in the full catalogue.
            minimum_games_per_year = 200

            chart_stats = yearly_stats[
                yearly_stats["games"] >= minimum_games_per_year
            ].copy()

            if not chart_stats.empty:
                fig = px.line(
                    chart_stats,
                    x="release_year",
                    y="flagged_hit_rate",
                    markers=True,
                    custom_data=["games"],
                    labels={
                        "release_year": "Release year",
                        "flagged_hit_rate": "Share flagged HIT",
                    },
                )

                fig.update_traces(
                    line=dict(
                        color="#818cf8",
                        width=3,
                    ),
                    marker=dict(size=6),
                    hovertemplate=(
                        "Release year: %{x}<br>"
                        "Flagged HIT: %{y:.2%}<br>"
                        "Games in year: %{customdata[0]:,}"
                        "<extra></extra>"
                    ),
                )

                fig.update_layout(
                    yaxis_tickformat=".0%",
                    margin=dict(l=10, r=10, t=20, b=10),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

                st.caption(
                    f"Shows years with at least {minimum_games_per_year:,} "
                    "catalogue games. Years with fewer games are omitted "
                    "from this chart only; they remain in the catalogue. "
                    "Rates represent model classifications, not true "
                    "historical success rates. Hover for game counts."
                )
            else:
                st.info(
                    "No release years meet the minimum catalogue count of "
                    f"{minimum_games_per_year:,} games."
                )

        else:

            st.info("Release-year information is unavailable.")

    st.divider()

    st.subheader("Highest-ranked games")

    st.write(
        "These titles received the highest decision scores. "
        "Scores rank model outputs and are not probabilities."
    )

    top_games = predictions.nlargest(
        10,
        "decision_score",
    )

    display_columns = [
        "model_rank",
        "name",
        "release_year",
        "main_genre",
        "prediction_label",
        "decision_score",
        "steam_appid",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in top_games.columns
    ]

    st.dataframe(
        top_games[display_columns],
        hide_index=True,
        width="stretch",
    )

    st.markdown(
        """
        <div class="footnote">
            SteamPulse predicts a proxy for cumulative Steam
            recommendation engagement. It does not verify sales,
            revenue, or profitability.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2: EXPLORE CATALOGUE
# ============================================================

elif page == "Explore Catalogue":

    st.markdown(
        """
        <div class="eyebrow">DISCOVER</div>
        <div class="section-title">Explore the catalogue</div>
        <div class="section-subtitle">
            Search titles, filter the catalogue, and investigate how
            the model ranks individual games.
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search by title or Steam App ID",
        placeholder="Enter a title or numeric App ID...",
    )

    filter1, filter2, filter3 = st.columns(3)

    genres = sorted(
        predictions["main_genre"].unique().tolist()
    )

    categories = sorted(
        predictions["main_category"].unique().tolist()
    )

    with filter1:

        genre_choice = st.selectbox(
            "Genre",
            ["All genres"] + genres,
        )

    with filter2:

        category_choice = st.selectbox(
            "Category",
            ["All categories"] + categories,
        )

    with filter3:

        price_choice = st.selectbox(
            "Price model",
            [
                "All",
                "Free-to-play",
                "Paid",
                "Unknown",
            ],
        )

    filter4, filter5, filter6 = st.columns(3)

    with filter4:

        if minimum_year < maximum_year:

            year_range = st.slider(
                "Release year",
                min_value=minimum_year,
                max_value=maximum_year,
                value=(minimum_year, maximum_year),
            )

        else:

            year_range = (minimum_year, maximum_year)

            st.caption(f"Release year: {minimum_year}")

    with filter5:

        classification_choice = st.selectbox(
            "Model classification",
            [
                "All classifications",
                "Predicted HIT",
                "Not flagged as HIT",
            ],
        )

    with filter6:

        sort_choice = st.selectbox(
            "Sort results",
            [
                "Highest model score",
                "Model rank",
                "Newest releases",
                "Oldest releases",
                "Title A–Z",
            ],
        )

    filtered = predictions

    if search_query.strip():

        query = search_query.strip()

        title_match = filtered["name"].str.contains(
            query,
            case=False,
            regex=False,
            na=False,
        )

        appid_match = filtered["steam_appid"].astype(
            str
        ).str.contains(
            query,
            regex=False,
            na=False,
        )

        filtered = filtered[
            title_match | appid_match
        ]

    if genre_choice != "All genres":

        filtered = filtered[
            filtered["main_genre"] == genre_choice
        ]

    if category_choice != "All categories":

        filtered = filtered[
            filtered["main_category"] == category_choice
        ]

    if price_choice != "All":

        filtered = filtered[
            filtered["free_status"] == price_choice
        ]

    filtered = filtered[
        filtered["release_year"].between(
            year_range[0],
            year_range[1],
            inclusive="both",
        )
        | filtered["release_year"].isna()
    ]

    if classification_choice == "Predicted HIT":

        filtered = filtered[
            filtered["predicted_hit"] == 1
        ]

    elif classification_choice == "Not flagged as HIT":

        filtered = filtered[
            filtered["predicted_hit"] == 0
        ]

    if sort_choice == "Highest model score":

        filtered = filtered.sort_values(
            "decision_score",
            ascending=False,
        )

    elif sort_choice == "Model rank":

        filtered = filtered.sort_values(
            "model_rank",
            ascending=True,
        )

    elif sort_choice == "Newest releases":

        filtered = filtered.sort_values(
            "release_year",
            ascending=False,
            na_position="last",
        )

    elif sort_choice == "Oldest releases":

        filtered = filtered.sort_values(
            "release_year",
            ascending=True,
            na_position="last",
        )

    else:

        filtered = filtered.sort_values(
            "name",
            ascending=True,
        )

    st.divider()

    st.subheader(
        f"{len(filtered):,} games found"
    )

    page_size = st.selectbox(
        "Rows to display",
        [25, 50, 100, 200],
        index=1,
    )

    table_columns = [
        "model_rank",
        "name",
        "release_year",
        "main_genre",
        "main_category",
        "free_status",
        "prediction_label",
        "decision_score",
        "observed_outcome_status",
        "steam_appid",
    ]

    table_columns = [
        column
        for column in table_columns
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[table_columns].head(page_size),
        hide_index=True,
        width="stretch",
        height=550,
    )

    st.caption(
        f"Showing {min(page_size, len(filtered)):,} of "
        f"{len(filtered):,} matching games."
    )

    download_columns = [
        column
        for column in [
            "steam_appid",
            "name",
            "release_year",
            "main_genre",
            "main_category",
            "main_developer",
            "main_publisher",
            "free_status",
            "prediction_label",
            "decision_score",
            "observed_outcome_status",
            "model_rank",
        ]
        if column in filtered.columns
    ]

    download_data = filtered[
        download_columns
    ].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered games as CSV",
        data=download_data,
        file_name="steampulse_filtered_games.csv",
        mime="text/csv",
    )

    st.caption(
        "A missing observed outcome does not mean the game was "
        "unsuccessful. Decision scores are not probabilities."
    )


# ============================================================
# PAGE 3: GAME SPOTLIGHT
# ============================================================

elif page == "Game Spotlight":

    st.markdown(
        """
        <div class="eyebrow">GAME INTELLIGENCE</div>
        <div class="section-title">Game spotlight</div>
        <div class="section-subtitle">
            Explore an individual game's available metadata and
            model classification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Find a game",
        placeholder="Enter a title or Steam App ID...",
        key="spotlight_search",
    )

    if query.strip():

        matches = predictions[
            predictions["name"].str.contains(
                query,
                case=False,
                regex=False,
                na=False,
            )
            |
            predictions["steam_appid"].astype(str).str.contains(
                query,
                regex=False,
                na=False,
            )
        ].copy()

    else:

        matches = predictions.nlargest(
            100,
            "decision_score",
        ).copy()

        st.caption(
            "Showing highly ranked games. Search to find another title."
        )

    if matches.empty:

        st.warning(
            "No matching games found. Try another title or App ID."
        )

    else:

        matches = matches.nlargest(
            min(100, len(matches)),
            "decision_score",
        ).copy()

        matches["selector"] = (
            matches["name"]
            + " · App ID "
            + matches["steam_appid"].astype(str)
        )

        selected_label = st.selectbox(
            "Select a game",
            matches["selector"].tolist(),
        )

        selected = matches[
            matches["selector"] == selected_label
        ].iloc[0]

        st.divider()

        st.title(selected["name"])

        st.caption(
            f"Steam App ID: {int(selected['steam_appid'])}"
        )

        c1, c2, c3, c4 = st.columns(4)

        release_year = selected.get(
            "release_year",
            np.nan,
        )

        with c1:

            st.metric(
                "Release year",
                (
                    str(int(release_year))
                    if pd.notna(release_year)
                    else "Unknown"
                ),
            )

        with c2:

            st.metric(
                "Model rank",
                f"#{int(selected['model_rank']):,}",
            )

        with c3:

            st.metric(
                "Decision score",
                f"{selected['decision_score']:.4f}",
            )

        with c4:

            st.metric(
                "Classification",
                (
                    "Predicted HIT"
                    if selected["predicted_hit"] == 1
                    else "Not flagged"
                ),
            )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("Game metadata")

            metadata = {
                "Genre": selected["main_genre"],
                "Category": selected["main_category"],
                "Developer": selected["main_developer"],
                "Publisher": selected["main_publisher"],
                "Price model": selected["free_status"],
                "Steam App ID": int(selected["steam_appid"]),
            }

            metadata_df = pd.DataFrame(
                {
                    "Attribute": list(metadata.keys()),
                    "Value": [
                        str(value)
                        for value in metadata.values()
                    ],
                }
            )

            st.dataframe(
                metadata_df,
                hide_index=True,
                width="stretch",
            )

        with right:

            st.subheader("Observed outcome")

            observed_status = selected[
                "observed_outcome_status"
            ]

            if observed_status == "Not observed":

                st.info(
                    "No observed target label is available for "
                    "this game in our dataset."
                )

                st.write(
                    "The absence of an observed label does not "
                    "mean the game was unsuccessful."
                )

            else:

                st.write(f"**{observed_status}**")

                st.caption(
                    "This label uses our recommendation threshold, "
                    "not verified sales or profit."
                )

        st.divider()

        st.subheader("Understanding the prediction")

        threshold = float(
            model_metrics.get(
                "threshold",
                0.0191066154,
            )
        )

        if selected["predicted_hit"] == 1:

            st.success(
                "The model's decision score meets or exceeds "
                "the selected HIT threshold."
            )

        else:

            st.info(
                "The model's decision score is below the selected "
                "HIT threshold."
            )

        st.write(
            f"Decision score: **{selected['decision_score']:.6f}**"
        )

        st.write(
            f"Decision threshold: **{threshold:.6f}**"
        )

        st.warning(
            "The decision score is not a probability. The "
            "classification does not establish commercial success."
        )


# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        """
        <div class="eyebrow">MODEL CARD</div>
        <div class="section-title">Performance & methodology</div>
        <div class="section-subtitle">
            Understand what SteamPulse predicts and how we evaluated it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Logistic Regression")

    c2.metric(
        "Regularization C",
        f"{model_metrics.get('C', 0.50):.2f}",
    )

    c3.metric("Input features", "45,518")

    st.divider()

    st.subheader("Validation performance")

    pr_auc = model_metrics.get(
        "validation_pr_auc",
        np.nan,
    )

    roc_auc = model_metrics.get(
        "validation_roc_auc",
        np.nan,
    )

    precision = model_metrics.get(
        "precision",
        np.nan,
    )

    recall = model_metrics.get(
        "recall",
        np.nan,
    )

    f1 = model_metrics.get(
        "f1",
        np.nan,
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "PR-AUC",
        f"{pr_auc:.4f}" if pd.notna(pr_auc) else "N/A",
    )

    metric_columns[1].metric(
        "ROC-AUC",
        f"{roc_auc:.4f}" if pd.notna(roc_auc) else "N/A",
    )

    metric_columns[2].metric(
        "Precision",
        f"{precision:.2%}" if pd.notna(precision) else "N/A",
    )

    metric_columns[3].metric(
        "Recall",
        f"{recall:.2%}" if pd.notna(recall) else "N/A",
    )

    metric_columns[4].metric(
        "F1",
        f"{f1:.4f}" if pd.notna(f1) else "N/A",
    )

    st.caption(
        "These are validation results. PR-AUC is below the original "
        "project target of 0.45."
    )

    st.divider()

    st.subheader("Target definition")

    st.write(
        """
        The positive target uses a recommendation-count threshold
        of **6,785 recommendations**, defined from the training
        distribution.
        """
    )

    st.write(
        """
        The target is a proxy for cumulative observed recommendation
        engagement. It is not a direct measurement of sales, revenue,
        profitability, or commercial success.
        """
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            **Training partition**

            - 13,369 labeled games
            - Release years 2000–2022
            - Used to fit preprocessing and the model
            """
        )

    with right:

        st.markdown(
            """
            **Validation partition**

            - 4,372 labeled games
            - Release years 2023–2024
            - Used for candidate comparison and threshold selection
            """
        )

    st.divider()

    st.subheader("Features used")

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            **Structured features**

            - Release year and month
            - Price-related metadata and free-to-play status
            - Platforms and platform requirements
            - Genres and categories
            - Developer and publisher metadata
            - Available languages
            """
        )

    with right:

        st.markdown(
            """
            **Text features**

            - Word n-grams from game descriptions
            - Character n-grams from game titles
            - Encoded genres and categories
            """
        )

    st.divider()

    st.subheader("Limitations")

    st.warning(
        """
        **Changing metadata:** Some fields may reflect information
        collected after a game's original release.

        **Unequal observation time:** Older games have had more
        time to accumulate recommendations.

        **Missing outcomes:** Missing labels are not treated
        as negative outcomes.

        **Validation exposure:** Multiple experiments used the
        same validation data, so the selected metric may be optimistic.
        Earlier experiments also evaluated the test set repeatedly,
        so it is not an untouched final evaluation.

        **Decision scores:** Model scores are not calibrated probabilities.
        """
    )

    st.divider()

    st.subheader("Catalogue coverage")

    c1, c2, c3 = st.columns(3)

    c1.metric("Games scored", f"{total_games:,}")
    c2.metric("Observed outcomes", f"{observed_count:,}")
    c3.metric("Unobserved outcomes", f"{unobserved_count:,}")

    st.info(
        "Catalogue predictions are model estimates, not a new "
        "independent evaluation of predictive performance."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.html(
    """
    <div class="footer">
        SteamPulse · Independent data science portfolio project<br>
        Model predictions are estimates, not verified commercial outcomes.
    </div>
    """
)