# Responsible AI Toolkit (Starter)

A [Streamlit](https://streamlit.io) dashboard to operationalize Responsible AI: load an audit checklist, edit Owner/Status, view KPIs and a risk heatmap, run lightweight drift checks (PSI), and align controls to the **NIST AI RMF**. Includes **demo mode** so everything is visible without proprietary data.

## Features
-  Editable checklist (Owner & Status)
-  KPIs + Status distribution
-  Risk heatmap (Owner × Status)
-  Lightweight drift check (PSI) *(optional; disabled if Evidently isn’t installed)*
-  NIST AI RMF alignment (upload `nist_mapping.csv`)
-  Export aligned/updated checklist (CSV/Excel)

## Quick start
```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python -m streamlit run app.py
