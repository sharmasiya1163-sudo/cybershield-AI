import streamlit as st
from pathlib import Path
import joblib

st.set_page_config(page_title="CyberShield AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top right,#102b45 0,#071426 45%,#050b15 100%); color:#e5eef8;}
h1,h2,h3 {color:#eaf5ff;}
[data-testid="stSidebar"] {background:#091a2c;}
.stButton>button {background:#0284c7;color:white;border:0;border-radius:9px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

MODEL = Path("models/logistic_regression_model.joblib")
VECTORIZER = Path("models/tfidf_vectorizer.joblib")

@st.cache_resource
def load_model():
    if MODEL.exists() and VECTORIZER.exists():
        return joblib.load(MODEL), joblib.load(VECTORIZER)
    return None, None

model, vectorizer = load_model()

st.title("🛡️ CyberShield AI")
st.subheader("AI-Powered Toxic Comment Detection")
st.caption("Analyze text to identify potentially harmful language.")
st.divider()

comment = st.text_area("Enter a comment to analyze", placeholder="Type or paste a comment...", height=150)

if st.button("🔍 Analyze Comment", use_container_width=True):
    if not comment.strip():
        st.warning("Please enter a comment.")
    elif model is None:
        st.error("Model files are not installed yet. Add the trained model and vectorizer to the models folder.")
    else:
        x = vectorizer.transform([comment])
        prediction = model.predict(x)[0]
        st.subheader("Analysis result")
        if int(prediction) == 1:
            st.error("Potentially toxic comment detected.")
        else:
            st.success("No toxicity detected by the model.")
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(x)[0]))
            st.write(f"Prediction confidence: {confidence:.1%}")
            st.progress(confidence)

st.divider()
st.caption("CyberShield AI | NLP project")
