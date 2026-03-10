# import streamlit as st
# import joblib
# import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt

# st.divider()
# st.subheader("📈 Feature Importance")

# feature_names = [
#     "prior_visits_30d",
#     "prior_spend_180d",
#     "treatment_exposed",
#     "impressions",
#     "clicks",
#     "conversion",
#     "revenue_usd",
#     "CTR",
#     "CPC",
#     "CONV_RATE"
# ]



# # Load trained model
# model = joblib.load("budget_model.pkl")

# st.set_page_config(page_title="AI Campaign Budget Predictor", layout="centered")

# st.title("📊 AI Campaign Budget Predictor")
# st.write("Predict the optimal campaign budget *before* launching an ad campaign.")

# st.divider()

# # -----------------------------
# # User Inputs
# # -----------------------------
# st.subheader("Enter Expected Campaign Details")

# prior_visits_30d = st.number_input(
#     "Prior Visits (Last 30 Days)",
#     min_value=0,
#     value=10
# )

# prior_spend_180d = st.number_input(
#     "Prior Spend (Last 180 Days) [USD]",
#     min_value=0.0,
#     value=100.0
# )

# treatment_exposed = st.selectbox(
#     "Treatment Exposed (Ad Shown Earlier)",
#     options=[0, 1],
#     format_func=lambda x: "Yes" if x == 1 else "No"
# )

# impressions = st.number_input(
#     "Expected Impressions",
#     min_value=0,
#     value=1000
# )

# clicks = st.number_input(
#     "Expected Clicks",
#     min_value=0,
#     value=50
# )

# conversion = st.number_input(
#     "Expected Conversions",
#     min_value=0,
#     value=5
# )

# revenue_usd = st.number_input(
#     "Expected Revenue [USD]",
#     min_value=0.0,
#     value=500.0
# )

# st.divider()


# importances = model.feature_importances_

# importance_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Importance": importances
# }).sort_values("Importance", ascending=False)

# st.bar_chart(importance_df.set_index("Feature"))
# # -----------------------------
# # Prediction Logic
# # -----------------------------
# if st.button("🚀 Predict Campaign Budget"):

#     # Feature Engineering (MUST match training)
#     CTR = clicks / (impressions + 1)
#     CPC = 1  # placeholder (same as training)
#     CONV_RATE = conversion / (clicks + 1)

#     input_data = np.array([[
#         prior_visits_30d,
#         prior_spend_180d,
#         treatment_exposed,
#         impressions,
#         clicks,
#         conversion,
#         revenue_usd,
#         CTR,
#         CPC,
#         CONV_RATE
#     ]])

#     prediction = model.predict(input_data)[0]

#     low = prediction * 0.9
#     high = prediction * 1.1

#     st.success(f"💰 Recommended Budget: **${prediction:,.2f}**")
#     st.info(f"📊 Suggested Range: ${low:,.2f} – ${high:,.2f}")


#     st.caption(
#         "This prediction is based on historical campaign performance and user behavior patterns."
#     )

# import matplotlib.pyplot as plt

# st.subheader("📉 Budget Visualization")

# fig, ax = plt.subplots()

# ax.bar(["Predicted Budget"], [prediction])

# st.pyplot(fig)


# st.divider()

# st.caption(
#     "⚙️ Powered by Random Forest Regression | Prototype for Hackathon Round-2"
# )

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# PAGE CONFIG (must be first)
# -----------------------------
st.set_page_config(page_title="AI Campaign Budget Predictor", layout="centered")


# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("budget_model.pkl")


# -----------------------------
# TITLE
# -----------------------------
st.title("📊 AI Campaign Budget Predictor")
st.write("Predict the optimal campaign budget *before* launching an ad campaign.")
st.divider()


# -----------------------------
# USER INPUTS
# -----------------------------
st.subheader("Enter Expected Campaign Details")

prior_visits_30d = st.number_input("Prior Visits (Last 30 Days)", 0, 1000, 10)
prior_spend_180d = st.number_input("Prior Spend (Last 180 Days) [USD]", 0.0, 100000.0, 100.0)

treatment_exposed = st.selectbox(
    "Treatment Exposed",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

impressions = st.number_input("Expected Impressions", 0, 1000000, 1000)
clicks = st.number_input("Expected Clicks", 0, 100000, 50)
conversion = st.number_input("Expected Conversions", 0, 10000, 5)
revenue_usd = st.number_input("Expected Revenue [USD]", 0.0, 1000000.0, 500.0)

st.divider()


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Campaign Budget"):

    # Feature engineering (must match training)
    CTR = clicks / (impressions + 1)
    CPC = 1
    CONV_RATE = conversion / (clicks + 1)

    input_data = np.array([[
        prior_visits_30d,
        prior_spend_180d,
        treatment_exposed,
        impressions,
        clicks,
        conversion,
        revenue_usd,
        CTR,
        CPC,
        CONV_RATE
    ]])

    prediction = model.predict(input_data)[0]

    low = prediction * 0.9
    high = prediction * 1.1

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.success(f"💰 Recommended Budget: **${prediction:,.2f}**")
    st.info(f"📊 Suggested Range: ${low:,.2f} – ${high:,.2f}")

    st.divider()


    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("📈 Feature Importance")

    feature_names = [
        "prior_visits_30d",
        "prior_spend_180d",
        "treatment_exposed",
        "impressions",
        "clicks",
        "conversion",
        "revenue_usd",
        "CTR",
        "CPC",
        "CONV_RATE"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))


    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    st.subheader("📉 Budget Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Predicted Budget"], [prediction])
    st.pyplot(fig)


    st.caption(
        "This prediction is based on historical campaign performance and user behavior patterns."
    )


st.divider()
st.caption("⚙️ Powered by Random Forest Regression | Hackathon Prototype")

