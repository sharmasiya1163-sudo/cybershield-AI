# CyberShield AI

CyberShield AI is an academic web application prototype for multi-label toxic comment detection.

## Features
- Cybersecurity-themed Streamlit interface
- Upload a labelled CSV dataset
- Train a TF-IDF + Logistic Regression multi-label classifier
- Predict six labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
- Display basic evaluation metrics
- Save trained model and vectorizer files locally

## Dataset format
The CSV must contain these columns:

`comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`

Each label should contain binary values: `0` or `1`.

## Run on Windows (VS Code terminal)

1. Open this folder in VS Code.
2. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

5. In the app, click **Load demonstration dataset** to test the interface, or upload your real labelled CSV and train on it.

## Important academic note
The included `sample_data.csv` is synthetic demonstration data used only to test the application workflow. It is not suitable for reporting model performance. Use your actual project dataset for training and evaluation.

## Model files
After training, the app creates:
- `models/toxicity_model.joblib`
- `models/tfidf_vectorizer.joblib`

These generated model files are local outputs. Do not commit large model files or private datasets to a public repository unless you have permission.
