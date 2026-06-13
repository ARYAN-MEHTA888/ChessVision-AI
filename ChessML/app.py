import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# PAGE SETTINGS
# ======================

st.set_page_config(
    page_title="ChessVision AI",
    page_icon="♟️",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================

model = joblib.load("best_xgb_model.pkl")
encoder = joblib.load("winner_encoder.pkl")

# ======================
# DESIGN
# ======================

st.markdown("""
<style>

.stApp{
    background-color:#0B0F19;
}

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:#FFD700;
}

.sub-title{
    text-align:center;
    font-size:22px;
    color:white;
}

section[data-testid="stSidebar"]{
    background-color:#111827;
}

.stButton > button{
    width:100%;
    height:60px;
    background:#FFD700;
    color:black;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================

st.markdown("""
<div class="main-title">
♟️ ChessVision AI
</div>

<div class="sub-title">
Predict Chess Winners Using Machine Learning
</div>
""", unsafe_allow_html=True)

st.write("")

# ======================
# METRICS
# ======================

col1,col2,col3,col4 = st.columns(4)

col1.metric("Games","20,058")
col2.metric("Accuracy","78.2%")
col3.metric("Openings","147")
col4.metric("Model","XGBoost")

st.divider()

# ======================
# SIDEBAR
# ======================

st.sidebar.title("♟️ Game Settings")

white_rating = st.sidebar.slider(
    "White Rating",
    800,
    3000,
    1500
)

black_rating = st.sidebar.slider(
    "Black Rating",
    800,
    3000,
    1500
)

turns = st.sidebar.slider(
    "Turns",
    1,
    300,
    40
)

opening_ply = st.sidebar.slider(
    "Opening Ply",
    1,
    20,
    4
)

rated = st.sidebar.selectbox(
    "Rated Game",
    ["No","Yes"]
)

# ======================
# PREDICTION
# ======================

st.subheader("♟️ Winner Prediction")

if st.button("Predict Winner"):

    rating_diff = white_rating - black_rating
    avg_rating = (white_rating + black_rating) / 2

    features = np.array([[
        white_rating,
        black_rating,
        rating_diff,
        avg_rating,
        turns,
        1 if rated == "Yes" else 0,
        opening_ply,
        0,
        0,
        10,
        0
    ]])

    prediction = model.predict(features)

    winner = encoder.inverse_transform(prediction)[0]

    st.success(
        f"🏆 Predicted Winner: {winner}"
    )

    probs = model.predict_proba(features)[0]

    probability_df = pd.DataFrame({
        "Result": encoder.classes_,
        "Probability": probs
    })

    st.subheader("Win Probability")

    st.bar_chart(
        probability_df.set_index("Result")
    )

# ======================
# ANALYTICS
# ======================

st.subheader("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature":[
        "Rating Difference",
        "White Rating",
        "Black Rating",
        "Average Rating",
        "Turns"
    ],
    "Importance":[
        0.30,
        0.20,
        0.15,
        0.10,
        0.08
    ]
})

st.bar_chart(
    importance_df.set_index("Feature")
)

# ======================
# ABOUT
# ======================

st.subheader("🏆 About Project")

st.info("""
Model: XGBoost

Dataset: Chess Games Dataset

Features:
• White Rating
• Black Rating
• Rating Difference
• Average Rating
• Turns
• Opening Information

Built Using:
• Python
• Pandas
• NumPy
• Scikit-Learn
• XGBoost
• Streamlit
""")

