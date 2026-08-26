import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Page configuration ---
st.set_page_config(
    page_title="Chicago Crime Analytics",
    page_icon=":material/local_police:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 2em;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }
    .result-card {
        background-color: #1a1d29;
        border-radius: 12px;
        padding: 1.5em;
        border: 1px solid #2d3142;
    }
    .metric-card {
        background-color: #1a1d29;
        border-radius: 10px;
        padding: 1.2em;
        border: 1px solid #2d3142;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8em;
        font-weight: 700;
        color: #f5f5f5;
    }
    .metric-card .label {
        color: #9ca3af;
        font-size: 0.85em;
        margin-top: 0.2em;
    }
    .header-wrap { text-align: center; padding-top: 0.5em; }
    .header-wrap h1 { margin-bottom: 0.1em; }
    .header-wrap p { color: #9ca3af; font-size: 0.95em; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Load model artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model_arrest_rf.pkl")
    le_crime_type = joblib.load("models/encoder_crime_type.pkl")
    return model, le_crime_type


model, le_crime_type = load_artifacts()


# --- Load and merge all data (crimes + weather + census) ---
@st.cache_data
def load_data():
    # Crimes
    df_crime = pd.read_csv("data/chicago_crimes_500k.csv", low_memory=False)
    df_crime["Date"] = pd.to_datetime(df_crime["Date"], errors="coerce")
    cols_crime = [
        "Date",
        "Primary Type",
        "Description",
        "Arrest",
        "Community Area",
        "Latitude",
        "Longitude",
    ]
    df_crime = df_crime[cols_crime].copy()
    df_crime = df_crime.drop_duplicates()

    df_crime["year"] = df_crime["Date"].dt.year
    df_crime["month"] = df_crime["Date"].dt.month
    df_crime["hour"] = df_crime["Date"].dt.hour
    df_crime["dayofweek"] = df_crime["Date"].dt.dayofweek
    df_crime["date_only"] = df_crime["Date"].dt.normalize()

    # Weather
    df_meteo = pd.read_csv("data/chicago_weather.csv")
    df_meteo = df_meteo.rename(columns={"time": "date_only"})
    df_meteo["date_only"] = pd.to_datetime(df_meteo["date_only"])

    df = df_crime.merge(df_meteo, on="date_only", how="left")

    # Census
    df_census = pd.read_csv("data/Census_Data_Chicago.csv")
    df = df.merge(
        df_census[
            [
                "Community Area Number",
                "COMMUNITY AREA NAME",
                "HARDSHIP INDEX",
                "PER CAPITA INCOME ",
            ]
        ],
        left_on="Community Area",
        right_on="Community Area Number",
        how="left",
    )

    return df, df_census


df, df_census = load_data()

# --- Header ---
st.markdown(
    """
    <div class="header-wrap">
        <h1>Chicago Crime Analytics</h1>
        <p>Exploring 500,000+ crime reports (2001–2025) combined with weather and socioeconomic data — with a live arrest prediction model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# --- Tabs ---
tab_predict, tab_insights, tab_about = st.tabs(["Predict", "City Insights", "About"])

with tab_predict:
    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("Incident details", divider="gray")

        crime_type_options = sorted(le_crime_type.classes_)
        primary_type = st.selectbox(
            "Crime type",
            options=crime_type_options,
            index=(
                crime_type_options.index("THEFT")
                if "THEFT" in crime_type_options
                else 0
            ),
        )

        neighborhood_options = sorted(
            df_census["COMMUNITY AREA NAME"].dropna().unique()
        )
        neighborhood = st.selectbox(
            "Neighborhood (Community Area)",
            options=neighborhood_options,
            index=0,
        )

        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour", 0, 23, 14)
            month = st.slider("Month", 1, 12, 7)
        with c2:
            dayofweek = st.selectbox(
                "Day of week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ][x],
            )
            year = st.number_input("Year", min_value=2001, max_value=2026, value=2024)

        c3, c4 = st.columns(2)
        with c3:
            temperature = st.slider("Mean temperature (°C)", -25.0, 40.0, 15.0)
        with c4:
            precipitation = st.number_input(
                "Precipitation (mm)", min_value=0.0, value=0.0, step=0.5
            )

        predict_btn = st.button(
            "Predict arrest likelihood",
            icon=":material/gavel:",
            use_container_width=True,
        )

    with col_result:
        st.subheader("Result", divider="gray")

        if not predict_btn:
            st.info(
                "Fill in the form on the left, then click **Predict arrest likelihood** to see the result here.",
                icon=":material/info:",
            )
        else:
            # --- 1. Retrieve socioeconomic data for the selected neighborhood ---
            census_row = df_census[
                df_census["COMMUNITY AREA NAME"] == neighborhood
            ].iloc[0]
            hardship = census_row["HARDSHIP INDEX"]
            income = census_row["PER CAPITA INCOME "]
            is_census_missing = 0  # a named neighborhood always has census data here

            # --- 2. Encode crime type ---
            primary_type_enc = le_crime_type.transform([primary_type])[0]

            # --- 3. Build feature vector, same order as training ---
            X_input = pd.DataFrame(
                [
                    {
                        "primary_type_enc": primary_type_enc,
                        "hour": hour,
                        "month": month,
                        "dayofweek": dayofweek,
                        "year": year,
                        "temperature_2m_mean (°C)": temperature,
                        "precipitation_sum (mm)": precipitation,
                        "HARDSHIP INDEX": hardship,
                        "PER CAPITA INCOME ": income,
                        "is_census_missing": is_census_missing,
                    }
                ]
            )

            # --- 4. Predict ---
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]
            arrest_proba = proba[1] * 100

            label = "Arrest likely" if prediction == 1 else "Arrest unlikely"
            color = "#4ade80" if prediction == 1 else "#f87171"

            st.markdown(
                f"""
            <div class="result-card">
                <p style="color:#9ca3af; margin-bottom:4px; font-size:0.9em;">PREDICTION</p>
                <p style="font-size:2em; font-weight:700; margin:0; color:{color};">{label}</p>
                <p style="color:#a78bfa; margin-top:4px;">Estimated probability: {arrest_proba:.1f}%</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown(f"**Context: {neighborhood}**")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""<div class="metric-card"><div class="value">{hardship:.0f}</div><div class="label">Hardship Index</div></div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-card"><div class="value">${income:,.0f}</div><div class="label">Per capita income</div></div>""",
                    unsafe_allow_html=True,
                )

with tab_insights:
    import plotly.express as px

    # --- Key metrics ---
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (f"{len(df):,}", "Total incidents"),
        (f"{df['Arrest'].mean()*100:.1f}%", "Overall arrest rate"),
        (f"{df['temperature_2m_mean (°C)'].mean():.1f}°C", "Avg. temperature"),
        (f"{df['year'].min()}–{df['year'].max()}", "Period covered"),
    ]
    for col, (value, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    st.write("")

    # --- Top crimes + arrest rate by crime type ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top 10 crime types**")
        top_crimes = df["Primary Type"].value_counts().head(10).reset_index()
        top_crimes.columns = ["Crime type", "Count"]
        fig1 = px.bar(top_crimes, x="Count", y="Crime type", orientation="h",
                      color="Count", color_continuous_scale="Blues")
        fig1.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False,
                            template="plotly_dark", coloraxis_showscale=False,
                            height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown("**Arrest rate by crime type** (top 10 by volume)")
        arrest_by_type = (
            df[df["Primary Type"].isin(top_crimes["Crime type"])]
            .groupby("Primary Type")["Arrest"].mean().sort_values(ascending=False).reset_index()
        )
        arrest_by_type.columns = ["Crime type", "Arrest rate"]
        fig2 = px.bar(arrest_by_type, x="Arrest rate", y="Crime type", orientation="h",
                      color="Arrest rate", color_continuous_scale="Purples")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False,
                            template="plotly_dark", coloraxis_showscale=False,
                            height=400, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Hour x Month heatmap ---
    st.markdown("**Crime volume by hour and month**")
    heatmap_data = df.groupby(["hour", "month"]).size().reset_index(name="count")
    heatmap_pivot = heatmap_data.pivot(index="hour", columns="month", values="count")
    fig3 = px.imshow(heatmap_pivot, color_continuous_scale="Blues", aspect="auto",
                      labels=dict(x="Month", y="Hour", color="Incidents"))
    fig3.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # --- Temperature effect + socioeconomic effect ---
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Crimes by temperature range**")
        bins = list(range(-25, 41, 5))
        labels = [f"{b}–{b+5}°C" for b in bins[:-1]]
        daily = df.groupby("date_only").agg(
            nb_crimes=("Primary Type", "count"),
            temp=("temperature_2m_mean (°C)", "mean"),
        ).dropna()
        daily["tranche"] = pd.cut(daily["temp"], bins=bins, labels=labels)
        temp_effect = daily.groupby("tranche", observed=True)["nb_crimes"].mean().reset_index()
        fig4 = px.line(temp_effect, x="tranche", y="nb_crimes", markers=True)
        fig4.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Temperature range", yaxis_title="Avg. crimes/day")
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.markdown("**Hardship Index vs. daily crime volume** (by neighborhood)")
        by_neighborhood = df.groupby("COMMUNITY AREA NAME").agg(
            nb_crimes=("Primary Type", "count"),
            hardship=("HARDSHIP INDEX", "first"),
        ).dropna().reset_index()
        fig5 = px.scatter(by_neighborhood, x="hardship", y="nb_crimes",
                           hover_name="COMMUNITY AREA NAME",
                           trendline="ols", color_discrete_sequence=["#a78bfa"])
        fig5.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Hardship Index", yaxis_title="Total crimes (2001–2025)")
        st.plotly_chart(fig5, use_container_width=True)

with tab_about:
    st.markdown("""
    ### About this project

    This dashboard is built on 500,000+ Chicago crime reports (2001–2025), combined with
    daily weather data and neighborhood-level socioeconomic indicators from the US Census.

    **Data sources:**
    - Chicago Police Department — crime reports (via Chicago Data Portal)
    - Open-Meteo — historical daily weather for Chicago
    - US Census Bureau — Chicago community area socioeconomic indicators

    **Model:** Random Forest classifier predicting arrest likelihood, trained with
    `class_weight="balanced"` to better capture the minority class (arrests represent
    ~25% of incidents). See the notebook for full methodology, evaluation metrics,
    and discussion of trade-offs (precision vs. recall).

    **Key finding:** the crime type is by far the strongest predictor of arrest likelihood
    (≈70% of feature importance), well ahead of time, weather, or neighborhood factors.

    **Limitations:**
    - The model reflects patterns in *reported and recorded* crimes, not actual crime rates
    - Socioeconomic and arrest data can reflect systemic biases in policing, not just crime severity
    - Sample of 500,000 rows out of a much larger full dataset
    """)

    st.divider()
    st.markdown("### Try it yourself — example scenarios for the Predict tab")

    scenarios = pd.DataFrame([
        {"Scenario": "Near-automatic arrest", "Crime type": "NARCOTICS", "Hour": 22, "Notes": "Possession is often caught in the act"},
        {"Scenario": "Rarely solved on the spot", "Crime type": "THEFT", "Hour": 15, "Notes": "Usually discovered after the fact, no immediate suspect"},
        {"Scenario": "Violent crime, high arrest expectation", "Crime type": "WEAPONS VIOLATION", "Hour": 2, "Notes": "Test at night vs. daytime"},
        {"Scenario": "Isolating the weather effect", "Crime type": "BATTERY", "Hour": 14, "Notes": "Compare -20°C vs 35°C, everything else fixed"},
        {"Scenario": "Isolating the neighborhood effect", "Crime type": "THEFT", "Hour": 15, "Notes": "Compare a high-Hardship-Index vs. low-Hardship-Index neighborhood"},
    ])

    st.dataframe(scenarios, hide_index=True, use_container_width=True)

    st.caption(
        "Tip: compare the predicted probabilities across these scenarios rather than reading "
        "each one in isolation — the relative differences are what reveal whether the model "
        "learned something sensible."
    )
