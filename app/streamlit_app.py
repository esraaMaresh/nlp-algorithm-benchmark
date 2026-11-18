import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="NLP Algorithm Benchmark",
    layout="wide"
)

# Load warm CSS
def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Go up one level from 'app' to 'nlp_algorithm-benchmark', then into 'assets/style.css'
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")

if os.path.exists(css_path):
    load_css(css_path)
else:
    st.warning(f"CSS file not found at: {css_path}")
# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span class="material-symbols-rounded" style="font-size: 36px; vertical-align: -6px; margin-right: 8px;">science</span>
        NLP Algorithm Benchmark Arena
    </div>
    <div class="hero-subtitle">
        Train, compare, and stress-test 3 classical ML algorithms head-to-head on real-world sentiment classification.
    </div>
</div>
""", unsafe_allow_html=True)

# Load saved models
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    nb = joblib.load(os.path.join(models_dir, "naive_bayes.joblib"))
    lr = joblib.load(os.path.join(models_dir, "logistic_regression.joblib"))
    svm = joblib.load(os.path.join(models_dir, "svm.joblib"))
    metrics = joblib.load(os.path.join(models_dir, "benchmark_metrics.joblib"))
    return vectorizer, nb, lr, svm, metrics

vectorizer, nb_model, lr_model, svm_model, benchmark_metrics = load_models()

ALGO_MAP = {
    "Multinomial Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "Support Vector Machine (LinearSVC)": svm_model
}

# =============================================
# SECTION 1: Live Prediction Arena
# =============================================
st.markdown("---")
st.subheader(":material/forum: Live Prediction Arena")

user_text = st.text_input(
    "Type any sentence to classify:",
    placeholder='e.g. "This product is absolutely terrible, worst purchase ever"'
)

if user_text:
    text_vec = vectorizer.transform([user_text])
    cols = st.columns(3, gap="medium")

    for idx, (name, model) in enumerate(ALGO_MAP.items()):
        with cols[idx]:
            pred = model.predict(text_vec)[0]
            proba = model.predict_proba(text_vec)[0]

            label = "Positive" if pred == 1 else "Negative"
            confidence = proba[pred] * 100
            card_class = "algo-card-positive" if pred == 1 else "algo-card-negative"
            pill_class = "pill-positive" if pred == 1 else "pill-negative"
            icon_name = "thumb_up" if pred == 1 else "thumb_down"

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-weight: 700; font-size: 1rem; margin-bottom: 10px; color: #fff8ee;">{name}</div>
                <div class="{pill_class}">
                    <span class="material-symbols-rounded">{icon_name}</span> {label}
                </div>
                <div style="margin-top: 10px; color: rgba(255,248,238,0.7); font-size: 0.85rem;">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(proba[1], text=f"Positive: {proba[1]*100:.1f}%")

# =============================================
# SECTION 2: Benchmark Comparison Table
# =============================================
st.markdown("---")
st.subheader(":material/leaderboard: Algorithm Benchmark Comparison")

comparison_data = []
for name, data in benchmark_metrics.items():
    report = data["report"]
    comparison_data.append({
        "Algorithm": name,
        "Accuracy": f"{data['accuracy'] * 100:.2f}%",
        "Precision (Pos)": f"{report['1']['precision'] * 100:.1f}%",
        "Recall (Pos)": f"{report['1']['recall'] * 100:.1f}%",
        "F1-Score (Pos)": f"{report['1']['f1-score'] * 100:.1f}%",
        "Precision (Neg)": f"{report['0']['precision'] * 100:.1f}%",
        "Recall (Neg)": f"{report['0']['recall'] * 100:.1f}%",
        "F1-Score (Neg)": f"{report['0']['f1-score'] * 100:.1f}%",
    })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# =============================================
# SECTION 3: Confusion Matrices
# =============================================
st.markdown("---")
st.subheader(":material/grid_on: Confusion Matrices")

cm_cols = st.columns(3, gap="medium")

for idx, (name, data) in enumerate(benchmark_metrics.items()):
    with cm_cols[idx]:
        st.markdown(f"**{name}**")
        cm = data["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Pred Negative", "Pred Positive"]
        )
        st.dataframe(cm_df, use_container_width=True)

# =============================================
# SECTION 4: Top Predictive Words
# =============================================
st.markdown("---")
st.subheader(":material/text_snippet: Top Predictive Words (Logistic Regression Coefficients)")
st.caption("Which words most strongly push the prediction toward Positive or Negative?")

feature_names = vectorizer.get_feature_names_out()
lr_coefficients = lr_model.coef_[0]
top_positive_indices = np.argsort(lr_coefficients)[-10:][::-1]
top_negative_indices = np.argsort(lr_coefficients)[:10]

fp_col, fn_col = st.columns(2, gap="large")

with fp_col:
    st.markdown("#### :material/trending_up: Top Positive Signal Words")
    for i in top_positive_indices:
        st.markdown(f"""
        <div class="word-card" style="border-left-color: #fbbf24;">
            <span style="font-weight: 700; color: #fbbf24;">{feature_names[i]}</span>
            <span style="float: right; color: rgba(255,248,238,0.6);">coeff: {lr_coefficients[i]:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

with fn_col:
    st.markdown("#### :material/trending_down: Top Negative Signal Words")
    for i in top_negative_indices:
        st.markdown(f"""
        <div class="word-card" style="border-left-color: #f97316;">
            <span style="font-weight: 700; color: #f97316;">{feature_names[i]}</span>
            <span style="float: right; color: rgba(255,248,238,0.6);">coeff: {lr_coefficients[i]:.3f}</span>
        </div>
        """, unsafe_allow_html=True)