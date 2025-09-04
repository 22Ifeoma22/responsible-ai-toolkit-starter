
# Responsible AI Toolkit (Starter)

This starter contains a Streamlit dashboard showing:
- SHAP explainability
- Fairlearn fairness metrics
- Evidently drift & classification quality

## How to run
1) Create and activate a virtual environment
   - Windows (PowerShell):
     ```ps1
     python -m venv .venv
     .\.venv\Scripts\Activate
     ```
   - macOS/Linux (bash):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2) Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the app
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser if it doesn’t open automatically.
