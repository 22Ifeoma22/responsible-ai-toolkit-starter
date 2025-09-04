
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import streamlit as st
from pathlib import Path

from pathlib import Path
import pandas as pd
import streamlit as st
# --- Evidently guard so the app never crashes ---
EVIDENTLY_OK = True
EVIDENTLY_ERR = None
# --- Evidently (optional) ---
EVIDENTLY_OK = True
EVIDENTLY_ERR = None
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ClassificationQualityMetric
except Exception as e:
    EVIDENTLY_OK = False
    EVIDENTLY_ERR = e

# ---- Config for the checklist file (no hard-coded C:\ paths) ----
DATA_PATH = Path(__file__).parent / "ISO42001_NIST_Audit_Checklist_Dashboard.xlsx"
SHEET = 0  # or "Sheet1" if your Excel sheet has a name

def load_data(upload=None):
    """Try uploaded file, then local file; otherwise show demo data."""
    if upload is not None:
        return pd.read_excel(upload, sheet_name=SHEET)

    if DATA_PATH.exists():
        return pd.read_excel(DATA_PATH, sheet_name=SHEET)

    # Fallback demo so the app never crashes
    demo = pd.DataFrame({
        "Clause": ["AIMS-6", "AIMS-7", "AIMS-8"],
        "Control": ["Risk Assessment", "Roles & Responsibilities", "Monitoring"],
        "Status": ["In Progress", "Complete", "Not Started"],
        "Owner": ["Model Risk", "CISO", "Data Science"],
        "Evidence": ["risk_register.xlsx", "org_chart.pdf", "evidently_report.html"]
    })
    return demo

# ---- File uploader (appears in the app) ----
uploaded = st.file_uploader("Upload ISO42001/NIST checklist (.xlsx)", type=["xlsx"])
df = load_data(upload=uploaded)

st.subheader("📋 Audit Checklist Data")
st.dataframe(df)
# --- Demo helpers so everything is visible even without real data ---
st.markdown("---")
st.subheader("🧪 Demo helpers")
if st.button("Populate demo Owners/Status"):
    # make sure columns exist
    if status_col is None or status_col not in df.columns:
        df["Status"] = ""
        status_col = "Status"
    if owner_col is None or owner_col not in df.columns:
        df["Owner"] = ""
        owner_col = "Owner"

    # cast to string so the editor works
    df[status_col] = df[status_col].astype("string").fillna("")
    df[owner_col]  = df[owner_col].astype("string").fillna("")

    # fill demo values
    status_options = ["Not Started", "In Progress", "Complete", "Blocked"]
    owners = ["CISO", "Model Risk", "Data Science", "Product", "Legal"]
    df[status_col] = [status_options[i % len(status_options)] for i in range(len(df))]
    df[owner_col]  = [owners[i % len(owners)] for i in range(len(df))]

    st.success("Filled demo Owners/Status. KPIs, heatmap and NIST views below will update.")

st.markdown("---")
st.subheader("🔎 Filter & Progress")

# Make column names consistent (handles slightly different headers)
cols = {c.lower().strip(): c for c in df.columns}
get = lambda name: cols.get(name, None)

status_col = get("status")
owner_col  = get("owner") or get("accountable") or get("responsible")
id_col     = get("control id") or get("clause") or get("requirement id")
name_col   = get("control name") or get("requirement") or get("title")

# Filters
c1, c2, c3 = st.columns([1,1,1])
text_q    = c1.text_input("Search text (ID/Name/Question)", "")
status_sel = c2.multiselect("Status", sorted(df[status_col].dropna().unique()) if status_col else [])
owner_sel  = c3.multiselect("Owner",  sorted(df[owner_col].dropna().unique()) if owner_col else [])

f = df.copy()
if text_q:
    mask = pd.Series([False] * len(f))
    for col in [id_col, name_col, get("audit questions")]:
        if col and col in f:
            mask = mask | f[col].astype(str).str.contains(text_q, case=False, na=False)
    f = f[mask]

if status_col and status_sel:
    f = f[f[status_col].isin(status_sel)]
if owner_col and owner_sel:
    f = f[f[owner_col].isin(owner_sel)]

st.write("**Filtered view**")
st.dataframe(f, use_container_width=True)

# KPIs
total       = len(df)
complete    = int((df[status_col] == "Complete").sum()) if status_col else 0
in_prog     = int((df[status_col] == "In Progress").sum()) if status_col else 0
not_started = int((df[status_col] == "Not Started").sum()) if status_col else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total controls", total)
k2.metric("Complete", complete)
k3.metric("In Progress", in_prog)
k4.metric("Not Started", not_started)

# Status chart
if status_col:
    st.write("**Status distribution**")
    st.bar_chart(df[status_col].value_counts())
else:
    st.info("Add a 'Status' column to view progress charts.")

# Export filtered table
st.download_button(
    "⬇️ Download filtered table (CSV)",
    data=f.to_csv(index=False).encode("utf-8"),
    file_name="audit_filtered.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("🧭 Complete the checklist (Owner & Status)")

# 1) Ensure required columns exist
if status_col is None or status_col not in df.columns:
    df["Status"] = ""
    status_col = "Status"

if owner_col is None or owner_col not in df.columns:
    df["Owner"] = ""
    owner_col = "Owner"

# 2) One-click initialise if Status is empty
if df[status_col].isna().all() or (df[status_col] == "").all():
    if st.button("Set all Status to 'Not Started'"):
        df[status_col] = "Not Started"
        st.success("All Status set to 'Not Started'")

# 3) Quick editable view for just the key columns
show_cols = [c for c in [id_col, name_col, status_col, owner_col] if c]
st.caption("Tip: edit Status/Owner directly in this table, then export below.")
# nicer editing: dropdown for Status + free text Owner
status_options = ["Not Started", "In Progress", "Complete", "Blocked"]

show_cols = [c for c in [id_col, name_col, status_col, owner_col] if c]
# --- Make columns editable by casting to string ---
def ensure_str(colname):
    if colname and colname in df.columns:
        df[colname] = df[colname].astype("string").fillna("")

for col in [owner_col, status_col, id_col, name_col]:
    ensure_str(col)

status_options = ["Not Started", "In Progress", "Complete", "Blocked"]

# 🔽 EDITABLE TABLE
edited = st.data_editor(
    df[[c for c in [id_col, name_col, status_col, owner_col] if c]],
    use_container_width=True,
    num_rows="dynamic",
    key="editor_checklist",
    column_config={
        status_col: st.column_config.SelectboxColumn(
            "Status", options=status_options, help="Set progress status"
        ),
        owner_col: st.column_config.TextColumn("Owner", help="Control owner/team"),
    },
)

# Apply edits back & normalise for charts
df.loc[edited.index, [status_col, owner_col]] = edited[[status_col, owner_col]]
df[status_col] = (
    df[status_col]
    .fillna("")
    .replace({"not started": "Not Started", "in progress": "In Progress"}, regex=False)
)
df[owner_col] = df[owner_col].fillna("")

edited = st.data_editor(
    df[show_cols],
    use_container_width=True,
    num_rows="dynamic",
    key="editor_main",
    column_config={
        status_col: st.column_config.SelectboxColumn(
            "Status", options=status_options, help="Set progress status"
        ),
        owner_col: st.column_config.TextColumn("Owner", help="Control owner/team"),
    },
)

# apply edits back to df
df.loc[edited.index, [status_col, owner_col]] = edited[[status_col, owner_col]]

# clean values so charts behave
df[status_col] = (
    df[status_col]
    .fillna("")
    .replace({"not started": "Not Started", "in progress": "In Progress"}, regex=False)
)
df[owner_col] = df[owner_col].fillna("")

# Push edits back to df (match by index)
df.loc[edited.index, [status_col, owner_col]] = edited[[status_col, owner_col]]

# 4) Risk heatmap (Owner × Status)
st.markdown("### 🔥 Risk Heatmap (Owner × Status)")
if owner_col and status_col:
    pivot = pd.crosstab(df[owner_col].replace("", "Unassigned"),
                        df[status_col].replace("", "Unspecified"))
    st.dataframe(pivot, use_container_width=True)

    # Simple heatmap plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Status"); ax.set_ylabel("Owner")
        ax.set_title("Owner vs Status counts")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Heatmap plot unavailable here: {e}")

# 5) Export the updated checklist to Excel
from io import BytesIO
excel_buf = BytesIO()
try:
    df.to_excel(excel_buf, index=False, engine="openpyxl")
except Exception:
    # fallback if openpyxl not present
    df.to_excel(excel_buf, index=False)
st.download_button(
    "📦 Download UPDATED checklist (Excel)",
    data=excel_buf.getvalue(),
    file_name="ISO42001_NIST_Checklist_UPDATED.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Explainability
import shap

# Fairness
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

# Monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ClassificationQualityMetric

st.set_page_config(page_title="Responsible AI Toolkit", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def make_dummy_dataset(n_samples=3000, n_features=12, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.6, 0.4],
        flip_y=0.02,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["target"] = y

    # Create simple synthetic sensitive attributes
    rng = np.random.default_rng(random_state)
    df["gender"] = rng.choice(["female", "male"], size=n_samples, p=[0.48, 0.52])
    df["age_group"] = rng.choice(["under30", "30to50", "over50"], size=n_samples, p=[0.35, 0.5, 0.15])

    return df

@st.cache_data(show_spinner=False)
def train_model(df, test_size=0.3, random_state=42):
    X = df.drop(columns=["target"])
    y = df["target"]

    # Keep sensitive columns separate for fairness (not used in training here)
    sensitive_cols = ["gender", "age_group"]
    X_model = X.drop(columns=sensitive_cols)

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X_model, y, X[sensitive_cols], test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return clf, scaler, (X_train, X_test, y_train, y_test, sens_train, sens_test), metrics

def compute_fairness(y_true, y_pred, sensitive_series):
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series,
    )
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_series)
    try:
        # Need y_prob for proper EOD usually; we approximate with y_pred here
        eod_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_series)
    except Exception:
        eod_diff = np.nan

    return mf.by_group, dp_diff, eod_diff

def build_evidently_report(ref_df, cur_df, target_col="target", prob_col=None, pred_col=None):
    report = Report(metrics=[DataDriftPreset(), ClassificationQualityMetric(target=target_col, prediction=pred_col)])
    report.run(reference_data=ref_df, current_data=cur_df)
    return report

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Controls")
n_samples = st.sidebar.slider("Dataset size", min_value=1000, max_value=10000, value=3000, step=500)
test_size = st.sidebar.slider("Test size fraction", 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)
sensitive_attr = st.sidebar.selectbox("Sensitive attribute", ["gender", "age_group"])
st.sidebar.markdown("---")
st.sidebar.caption("Tip: switch the sensitive attribute to see fairness by different groups.")

# -----------------------------
# Main layout
# -----------------------------
st.title("🛡️ Responsible AI Toolkit: Explainability • Fairness • Drift • Compliance")

with st.spinner("Generating dataset and training model..."):
    df = make_dummy_dataset(n_samples=n_samples, random_state=random_state)
    clf, scaler, splits, perf = train_model(df, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test, sens_train, sens_test = splits

# Headline metrics
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{perf['accuracy']:.3f}")
m2.metric("F1 Score", f"{perf['f1']:.3f}")
m3.metric("ROC AUC", f"{perf['roc_auc']:.3f}")

st.markdown("---")

# -----------------------------
# Explainability (SHAP)
# -----------------------------
st.subheader("🔍 Explainability (SHAP)")

try:
    # TreeExplainer on a sample to keep it fast
    explainer = shap.TreeExplainer(clf)
    X_test_sample = X_test.sample(min(500, len(X_test)), random_state=random_state)
    X_test_sample_s = scaler.transform(X_test_sample)
    shap_values = explainer.shap_values(X_test_sample_s)

    st.write("**Global feature importance** (SHAP summary for positive class):")
    fig_summary = shap.summary_plot(
        shap_values[1] if isinstance(shap_values, list) else shap_values,
        X_test_sample,
        show=False
    )
    st.pyplot(bbox_inches="tight", dpi=120)

    st.caption("SHAP shows which features most influence predictions across the dataset.")
except Exception as e:
    st.info(f"SHAP could not render in this environment: {e}")

st.markdown("---")

# -----------------------------
# Fairness (Fairlearn)
# -----------------------------
st.subheader("⚖️ Fairness & Bias (Fairlearn)")
with st.spinner("Computing fairness by group..."):
    # Predictions for fairness
    y_pred = clf.predict(scaler.transform(X_test))
    # sensitive attribute (Series)
    s_series = sens_test[sensitive_attr]

    by_group_metrics, dp_diff, eod_diff = compute_fairness(y_test, y_pred, s_series)

st.write("**Performance by group**")
st.dataframe(by_group_metrics)

c1, c2 = st.columns(2)
c1.metric("Demographic Parity Difference", f"{dp_diff:.3f}")
c2.metric("Equalized Odds Difference", f"{eod_diff if not np.isnan(eod_diff) else 0:.3f}")

st.caption(
    "• Demographic Parity Difference near 0 indicates similar positive prediction rates across groups. "
    "• Equalized Odds Difference near 0 indicates similar error rates across groups."
)

st.markdown("---")

# -----------------------------
# Monitoring & Drift (Evidently)
# -----------------------------
st.subheader("📈 Monitoring & Drift (lightweight)")

import numpy as np

def psi(ref, cur, bins=10):
    """Population Stability Index for one numeric feature."""
    ref = np.asarray(ref, dtype=float); cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    edges = np.histogram_bin_edges(ref, bins=bins)
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r = r / max(r.sum(), 1); c = c / max(c.sum(), 1)
    r = np.clip(r, 1e-6, None); c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))

# Use training as reference and test as current
ref_df = X_train.copy()
cur_df = X_test.copy()

rows = []
for col in ref_df.columns:
    if np.issubdtype(ref_df[col].dtype, np.number):
        rows.append({"Feature": col, "PSI": round(psi(ref_df[col], cur_df[col]), 4)})

psi_table = pd.DataFrame(rows).sort_values("PSI", ascending=False)
st.write("**Simple drift (PSI)** — >0.10 small, >0.25 moderate, >0.50 major.")
st.dataframe(psi_table, use_container_width=True)



# -----------------------------
# Compliance mapping (read-only text)
# -----------------------------
st.subheader("📜 Compliance Mapping (Read-only)")
st.markdown(
    """
**ISO/IEC 42001**  
- AIMS 6–8: Risk assessment & controls → *Fairness metrics, drift monitoring, model performance*  
- AIMS 9–10: Monitoring & improvement → *Evidently reports & periodic reviews*

**NIST AI RMF 100-1**  
- Map → *Data & context understanding (drift)*  
- Measure → *SHAP, group metrics (Fairlearn)*  
- Manage → *Risk register & remediation plans (outside this demo)*

**GDPR Article 22**  
- Transparency & meaningful information: *Global SHAP explanations*  
- Non-discrimination: *Group parity checks*  
"""
)
