# CyberShield AI
AI-powered toxic comment detection web application.

## Planned features
- Toxic / non-toxic comment classification
- Six toxicity labels: toxic, severe toxic, obscene, threat, insult, identity hate
- Confidence scores
- Cybersecurity-themed Streamlit interface

## Tech stack
Python, Streamlit, scikit-learn, TF-IDF, Joblib, Pandas, NumPy.

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Model files required
After exporting your trained objects from the notebook, place:
- `models/logistic_regression_model.joblib`
- `models/tfidf_vectorizer.joblib`

This starter app is wired for a binary Logistic Regression model and one TF-IDF vectorizer. We will adapt it to your exact notebook objects and six-label setup once we inspect them.
