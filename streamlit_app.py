# =============================================================================
# LOAN BIAS DETECTION TOOL — STREAMLIT WEB APP (FIXED VERSION)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import google.generativeai as genai
import json
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Loan Bias Detection Tool",
    page_icon="⚖️",
    layout="wide"
)

# =============================================================================
# GEMINI SETUP
# =============================================================================

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_available = True
except Exception:
    gemini_available = False

# =============================================================================
# HEADER
# =============================================================================

st.title("⚖️ Loan Bias Detection Tool")
st.write("Detect hidden unfairness in loan approval decisions using statistics + AI")

if not gemini_available:
    st.warning("⚠️ Gemini API key not configured. AI explanations disabled.")

# =============================================================================
# FILE UPLOAD
# =============================================================================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

# =============================================================================
# LOAD & CLEAN DATA
# =============================================================================

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Basic cleaning
    df["Purpose"] = df["Purpose"].str.strip().str.title()
    df["Home Ownership"] = df["Home Ownership"].replace("HaveMortgage", "Home Mortgage")

    df["Credit Score"] = df["Credit Score"].fillna(df["Credit Score"].median())
    df["Annual Income"] = df["Annual Income"].fillna(df["Annual Income"].median())
    df["Bankruptcies"] = df["Bankruptcies"].fillna(0)
    df["Tax Liens"] = df["Tax Liens"].fillna(0)

    df = df.dropna(subset=["Years in current job"])

    return df

df = load_data(uploaded_file)

st.success(f"Dataset loaded: {len(df)} rows")

# =============================================================================
# APPROVAL RULE
# =============================================================================

threshold = st.slider("Credit Score Approval Threshold", 500, 850, 700)

df["Approved"] = (df["Credit Score"] >= threshold).astype(int)

total = len(df)
approved = df["Approved"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total", total)
col2.metric("Approved", approved)
col3.metric("Rejected", total - approved)

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["Credit Score"], bins=30)
axes[0].axvline(threshold)
axes[0].set_title("Credit Score")

axes[1].hist(df["Annual Income"], bins=30)
axes[1].set_title("Income")

st.pyplot(fig)
plt.close()

# =============================================================================
# AI FUNCTION
# =============================================================================

def get_ai_insight(label, stats_summary, di, p):
    if not gemini_available:
        return "AI unavailable", "Manual review required"

    prompt = f"""
    Analyze loan bias for {label}:
    {stats_summary}
    DI={di}, p={p}

    Return JSON:
    {{"explanation":"...","recommendation":"..."}}
    """

    try:
        response = gemini_model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "")
        data = json.loads(clean)
        return data.get("explanation"), data.get("recommendation")
    except:
        return "AI error", "Manual review"

# =============================================================================
# BIAS ANALYSIS
# =============================================================================

def analyze(df, col):
    stats_df = df.groupby(col)["Approved"].agg(["count", "sum"]).reset_index()
    stats_df["rate"] = stats_df["sum"] / stats_df["count"]

    max_r = stats_df["rate"].max()
    min_r = stats_df["rate"].min()

    di = min_r / max_r if max_r > 0 else 1

    cont = pd.crosstab(df[col], df["Approved"])
    if cont.shape[0] > 1:
        chi2, p, _, _ = stats.chi2_contingency(cont)
    else:
        p = 1

    explanation, rec = get_ai_insight(col, stats_df.to_string(), di, p)

    return stats_df, di, p, explanation, rec

# =============================================================================
# RUN ANALYSIS
# =============================================================================

columns = ["Home Ownership", "Purpose", "Years in current job", "Term"]

results = {}

for col in columns:
    try:
        res = analyze(df, col)
        results[col] = res
    except Exception as e:
        st.error(f"Error in {col}: {e}")

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

for col, (stats_df, di, p, explanation, rec) in results.items():

    st.subheader(col)

    fig, ax = plt.subplots()
    ax.bar(stats_df[col], stats_df["rate"])
    ax.set_title(f"{col} Approval Rate")
    st.pyplot(fig)
    plt.close()

    st.write(f"Disparate Impact: {di:.2f}")
    st.write(f"P-value: {p:.4f}")

    if di < 0.8:
        st.error("⚠️ Bias Detected")
    else:
        st.success("✅ Fair")

    st.info(f"AI: {explanation}")
    st.info(f"Recommendation: {rec}")

# =============================================================================
# DOWNLOAD RESULTS
# =============================================================================

output = io.BytesIO()

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    for col, (stats_df, _, _, _, _) in results.items():
        stats_df.to_excel(writer, sheet_name=col[:31], index=False)

st.download_button(
    label="Download Excel Report",
    data=output.getvalue(),
    file_name="bias_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("Built with Streamlit + Gemini AI")