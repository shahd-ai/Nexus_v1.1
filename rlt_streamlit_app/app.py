import sys
import os
import pandas as pd
import streamlit as st

# Allow importing script.py
sys.path.append("..")

from Enhanced_rlt import train_and_save, load_and_predict

MODEL_PATH = "../rlt_models/streamlit_rlt_model.pkl"

st.set_page_config(page_title="RLT Auto ML", layout="wide")

# Custom CSS for cleaner look
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    h1 {
        color: #1f2937;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #4b5563;
        font-size: 1.5rem;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌳 Reinforcement Learning Tree – Auto ML")

# =====================================================
# 1️⃣ Upload Dataset
# =====================================================
st.header("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset loaded successfully")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes("number").columns))
    
    st.write("**Preview:**")
    st.dataframe(df.head(), use_container_width=True)

    # =====================================================
    # 2️⃣ Select Target
    # =====================================================
    st.header("2️⃣ Select Target Column")

    target_col = st.selectbox("Target column", df.columns)

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            metrics = train_and_save(
                df=df,
                target_col=target_col,
                model_path=MODEL_PATH
            )

        st.success("✅ Model trained successfully!")
        
        st.subheader("📊 Model Metrics")
        
        # Display metrics in columns
        metric_cols = st.columns(len(metrics))
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            with metric_cols[idx]:
                st.metric(
                    metric_name.upper(),
                    f"{metric_value:.4f}" if isinstance(metric_value, float) else metric_value
                )

        # Save feature names for prediction
        st.session_state["features"] = (
            df.drop(columns=[target_col])
            .select_dtypes("number")
            .columns.tolist()
        )

        st.session_state["trained"] = True

# =====================================================
# 3️⃣ Manual Prediction
# =====================================================
if st.session_state.get("trained", False):
    st.header("3️⃣ Manual Prediction")

    features = st.session_state["features"]

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f"
            )

    if st.button("🎯 Predict"):
        input_df = pd.DataFrame([input_data])

        result = load_and_predict(MODEL_PATH, input_df)

        st.subheader("🎉 Prediction Result")

        if isinstance(result, tuple):
            preds, probs = result
            
            st.success(f"**Predicted Class:** {preds[0]}")
            
            if probs is not None:
                st.write("**Class Probabilities:**")
                prob_df = pd.DataFrame([probs[0]], columns=[f"Class {i}" for i in range(len(probs[0]))])
                st.dataframe(prob_df, use_container_width=True)
        else:
            st.success(f"**Predicted Value:** {result[0]:.4f}")