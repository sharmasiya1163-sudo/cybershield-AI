import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COL = "comment_text"

st.set_page_config(page_title="CyberShield AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 15% 10%, #10243a 0, #080d18 42%, #050812 100%); color:#e8f1ff;}
section[data-testid="stSidebar"] {background:#0b1525; border-right:1px solid #1b4560;}
h1,h2,h3 {color:#eaf6ff !important;}
div[data-testid="stMetric"] {background:#101e31; border:1px solid #1e5870; padding:14px; border-radius:12px;}
.stButton>button {background:#00c2a8; color:#031018; border:0; border-radius:8px; font-weight:700;}
.stButton>button:hover {background:#4ce5d0; color:#031018;}
div[data-testid="stTextArea"] textarea {background:#0c1728; color:#f3f7ff; border:1px solid #24516b;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🛡️ CyberShield AI</h1>", unsafe_allow_html=True)
st.markdown("### Multi-label toxic comment detection")
st.write("Analyze a comment for toxicity categories using a TF-IDF + Logistic Regression machine-learning pipeline.")

with st.sidebar:
    st.markdown("## ⚙️ Model setup")
    st.write("Upload your labelled CSV dataset, or load the included demonstration dataset.")
    uploaded = st.file_uploader("Upload training CSV", type=["csv"])
    use_demo = st.button("Load demonstration dataset", use_container_width=True)
    st.caption("Required columns: comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate.")
    st.divider()
    st.caption("Model: TF-IDF (unigrams + bigrams) and one Logistic Regression classifier per label.")

def prepare_data(data):
    required = [TEXT_COL] + LABELS
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = data[required].copy()
    data[TEXT_COL] = data[TEXT_COL].fillna("").astype(str)
    for label in LABELS:
        data[label] = pd.to_numeric(data[label], errors="coerce").fillna(0).astype(int)
        data[label] = data[label].clip(0, 1)
    data = data[data[TEXT_COL].str.strip().ne("")]
    return data

if use_demo:
    demo_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    if os.path.exists(demo_path):
        st.session_state["training_df"] = pd.read_csv(demo_path)
        st.session_state["data_name"] = "Included synthetic demonstration data"
    else:
        st.error("sample_data.csv was not found next to app.py.")

if uploaded is not None:
    try:
        st.session_state["training_df"] = pd.read_csv(uploaded)
        st.session_state["data_name"] = uploaded.name
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

df = st.session_state.get("training_df")

if df is None:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("## Welcome to your comment safety dashboard")
        st.write("Start by loading the included demo dataset or uploading your own labelled dataset. The app will train a model and let you test comments.")
        st.info("The demonstration dataset is synthetic and is only for checking that the application runs. Use your real project dataset for meaningful results.")
    with right:
        st.markdown("## Expected dataset format")
        st.code("comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate\\nThis is an example,0,0,0,0,0,0", language="csv")
    st.stop()

try:
    clean_df = prepare_data(df)
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown("## Dataset overview")
a, b, c = st.columns(3)
a.metric("Comments", f"{len(clean_df):,}")
b.metric("Toxicity labels", len(LABELS))
c.metric("Source", st.session_state.get("data_name", "Uploaded CSV"))
with st.expander("Preview dataset"):
    st.dataframe(clean_df.head(10), use_container_width=True)

st.markdown("## Train model")
max_rows = min(len(clean_df), 50000)
default_rows = min(len(clean_df), 20000)
train_rows = st.slider("Rows to use for training", min_value=min(20, max_rows), max_value=max_rows,
                       value=max(20, default_rows), step=1 if max_rows <= 1000 else 100)
if st.button("🚀 Train CyberShield AI", type="primary"):
    work = clean_df.sample(n=train_rows, random_state=42) if train_rows < len(clean_df) else clean_df.copy()
    X = work[TEXT_COL]
    y = work[LABELS]
    invalid = [label for label in LABELS if y[label].nunique() < 2]
    if invalid:
        st.error("Each label must contain both 0 and 1 values. Labels needing more examples: " + ", ".join(invalid))
    else:
        with st.spinner("Training TF-IDF and Logistic Regression models..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            model = MultiOutputClassifier(
                LogisticRegression(max_iter=1000, class_weight="balanced")
            )
            model.fit(X_train_vec, y_train)
            predictions = model.predict(X_test_vec)
            st.session_state["vectorizer"] = vectorizer
            st.session_state["model"] = model
            st.session_state["metrics"] = {
                "accuracy": accuracy_score(y_test, predictions),
                "f1": f1_score(y_test, predictions, average="micro", zero_division=0),
                "test_rows": len(y_test)
            }
            os.makedirs("models", exist_ok=True)
            joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
            joblib.dump(model, "models/toxicity_model.joblib")
        st.success("Training complete. The model files were saved in the models folder.")

if "model" in st.session_state:
    st.markdown("## Model evaluation")
    m = st.session_state["metrics"]
    x, y, z = st.columns(3)
    x.metric("Exact-match accuracy", f"{m['accuracy']:.1%}")
    y.metric("Micro F1 score", f"{m['f1']:.1%}")
    z.metric("Test comments", f"{m['test_rows']:,}")

    st.markdown("## Analyze a comment")
    comment = st.text_area("Enter a comment to analyze", placeholder="Type or paste a comment here...", height=130)
    if st.button("Analyze comment"):
        if not comment.strip():
            st.warning("Please enter a comment first.")
        else:
            vec = st.session_state["vectorizer"].transform([comment])
            model = st.session_state["model"]
            pred = model.predict(vec)[0]
            st.markdown("### Analysis result")
            toxic_found = any(pred)
            if toxic_found:
                st.error("Potentially harmful content detected")
            else:
                st.success("No toxicity category detected by the model")
            result_cols = st.columns(3)
            for i, label in enumerate(LABELS):
                with result_cols[i % 3]:
                    st.metric(label.replace("_", " ").title(), "Detected" if pred[i] else "Not detected")
            st.caption("This is an automated prediction, not a definitive judgement. Review flagged content before taking action.")

    with st.expander("Save trained model files"):
        st.write("The training step creates `models/toxicity_model.joblib` and `models/tfidf_vectorizer.joblib` in your project folder.")
        st.code("models/toxicity_model.joblib\\nmodels/tfidf_vectorizer.joblib", language="text")

st.markdown("---")
st.caption("CyberShield AI • Academic prototype • TF-IDF + Logistic Regression")
