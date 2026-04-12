#  LOAN BIAS DETECTION TOOL
#  Detects hidden unfairness in credit/loan decision data
#
#  HOW TO RUN:
#    1. Install required libraries (only needed once):
#       pip install pandas openpyxl matplotlib seaborn scipy
#
#    2. Place your dataset file in the same folder as this script
#       (your file is named: credit_test.xlsx)
#
#    3. Run the script:
#       python bias_detection.py
#
#    4. A folder called "bias_report" will be created with all results.
# =============================================================================

import os
import sys                          # FIX 1: added missing sys import
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import google.generativeai as genai
import json

warnings.filterwarnings("ignore")  # Keep output clean for beginner

# =============================================================================
# API CONFIGURATION
# =============================================================================
# Replace with your actual API key from https://aistudio.google.com/
API_KEY = "GEMINI_API_KEY" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3-flash-preview')


# =============================================================================
# STEP 0: SETUP — Create output folder for results
# =============================================================================

OUTPUT_FOLDER = "bias_report"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("  LOAN BIAS DETECTION TOOL")
print("=" * 60)
print(f"\n Results will be saved to: ./{OUTPUT_FOLDER}/\n")


# =============================================================================
# STEP 1: LOAD THE DATA
# =============================================================================

print("[ Step 1 ] Loading dataset...")

FILE_PATH = r"C:\Users\Chiru\Downloads\credit_test1.csv"  # <-- Change this if your file has a different name

try:
    df = pd.read_csv(FILE_PATH)
    print(f"  Loaded {len(df):,} rows and {len(df.columns)} columns.\n")
except FileNotFoundError:
    print(f"\n  ERROR: Could not find '{FILE_PATH}'.")
    print(f"  Make sure the file is in the same folder as this script.\n")
    sys.exit()                      # FIX 2: replaced exit() with sys.exit()


# =============================================================================
# STEP 2: CLEAN THE DATA
# =============================================================================

print("[ Step 2 ] Cleaning data...")

# --- Fix inconsistent labels in Purpose column ---
# (e.g. 'other' and 'Other' should be the same thing)
df["Purpose"] = df["Purpose"].str.strip().str.title()

# --- Fix inconsistent Home Ownership labels ---
# 'HaveMortgage' is the same as 'Home Mortgage'
df["Home Ownership"] = df["Home Ownership"].replace("HaveMortgage", "Home Mortgage")

# --- Report missing values ---
missing = df.isnull().sum()
missing = missing[missing > 0]
print("\n  Missing values found:")
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"    - {col}: {count:,} missing ({pct:.1f}%)")

# --- Fill missing Credit Score and Annual Income with column median ---
# (Median is safer than average when data has outliers)
df["Credit Score"] = df["Credit Score"].fillna(df["Credit Score"].median())
df["Annual Income"] = df["Annual Income"].fillna(df["Annual Income"].median())

# --- Fill missing Bankruptcies and Tax Liens with 0 ---
df["Bankruptcies"] = df["Bankruptcies"].fillna(0)
df["Tax Liens"] = df["Tax Liens"].fillna(0)

# --- Drop rows still missing critical info ---
df = df.dropna(subset=["Years in current job"])

print(f"\n  After cleaning: {len(df):,} rows remain.\n")


# =============================================================================
# STEP 3: CREATE AN APPROVAL DECISION COLUMN
#
# Since there is no "Approved/Rejected" column in this dataset, we simulate
# a loan approval decision using a Credit Score threshold — a common
# real-world practice.
#
# Rule: Credit Score >= 700 → Approved (1)
#       Credit Score <  700 → Rejected (0)
#
# WHY THIS MATTERS FOR BIAS: If certain groups (e.g. renters vs homeowners)
# consistently have lower credit scores — even for non-financial reasons —
# this rule will systematically reject them more, which is a form of bias.
# =============================================================================

print("[ Step 3 ] Creating loan approval decisions (Credit Score >= 700 = Approved)...")

APPROVAL_THRESHOLD = 700
df["Approved"] = (df["Credit Score"] >= APPROVAL_THRESHOLD).astype(int)

total     = len(df)
approved  = df["Approved"].sum()
rejected  = total - approved
print(f"  Approved: {approved:,} ({approved/total*100:.1f}%)")
print(f"  Rejected: {rejected:,} ({rejected/total*100:.1f}%)\n")


# =============================================================================
# STEP 4: DATA OVERVIEW REPORT  (saved as a text file)
# =============================================================================

print("[ Step 4 ] Generating data overview report...")

with open(f"{OUTPUT_FOLDER}/1_data_overview.txt", "w") as f:
    f.write("LOAN BIAS DETECTION — DATA OVERVIEW\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Total Records : {len(df):,}\n")
    f.write(f"Approved      : {approved:,} ({approved/total*100:.1f}%)\n")
    f.write(f"Rejected      : {rejected:,} ({rejected/total*100:.1f}%)\n\n")

    f.write("COLUMN SUMMARY\n" + "-" * 30 + "\n")
    f.write(df.describe(include="all").to_string())
    f.write("\n\nMISSING VALUES (after cleaning)\n" + "-" * 30 + "\n")
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing) == 0:
        f.write("  None — all critical columns are clean.\n")
    else:
        f.write(remaining_missing.to_string())

print(f"  Saved: {OUTPUT_FOLDER}/1_data_overview.txt\n")


# =============================================================================
# STEP 5: VISUALISE KEY DISTRIBUTIONS
# =============================================================================

print("[ Step 5 ] Plotting key distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dataset Overview — Key Distributions", fontsize=16, fontweight="bold")

# --- Credit Score Distribution ---
axes[0, 0].hist(df["Credit Score"], bins=40, color="#4C72B0", edgecolor="white")
axes[0, 0].axvline(APPROVAL_THRESHOLD, color="red", linestyle="--", linewidth=2,
                   label=f"Approval Threshold ({APPROVAL_THRESHOLD})")
axes[0, 0].set_title("Credit Score Distribution")
axes[0, 0].set_xlabel("Credit Score")
axes[0, 0].set_ylabel("Number of Applicants")
axes[0, 0].legend()

# --- Annual Income Distribution ---
axes[0, 1].hist(df["Annual Income"].clip(upper=300000), bins=40,
                color="#55A868", edgecolor="white")
axes[0, 1].set_title("Annual Income Distribution (capped at $300k)")
axes[0, 1].set_xlabel("Annual Income ($)")
axes[0, 1].set_ylabel("Number of Applicants")

# --- Home Ownership Breakdown ---
home_counts = df["Home Ownership"].value_counts()
axes[1, 0].bar(home_counts.index, home_counts.values, color="#C44E52", edgecolor="white")
axes[1, 0].set_title("Applicants by Home Ownership")
axes[1, 0].set_xlabel("Home Ownership Status")
axes[1, 0].set_ylabel("Number of Applicants")
axes[1, 0].tick_params(axis="x", rotation=15)

# --- Loan Purpose Breakdown ---
purpose_counts = df["Purpose"].value_counts().head(8)
axes[1, 1].barh(purpose_counts.index, purpose_counts.values, color="#8172B2")
axes[1, 1].set_title("Top Loan Purposes")
axes[1, 1].set_xlabel("Number of Applicants")
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/2_data_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_FOLDER}/2_data_distributions.png\n")


# =============================================================================
# STEP 6: BIAS ANALYSIS FUNCTION
#
# For each "sensitive group" (e.g. Home Ownership type), we calculate:
#
#   1. APPROVAL RATE        — What % of each group got approved?
#
#   2. DISPARATE IMPACT     — Do some groups get approved far less than others?
#                             Industry standard: a ratio below 0.80 (80% rule)
#                             means the system is potentially discriminatory.
#                             Formula: (lowest group rate) / (highest group rate)
#
#   3. STATISTICAL TEST     — Is the difference real or just random noise?
#                             We use a Chi-Square test. If p < 0.05, the gap
#                             is statistically significant (likely real bias).
# =============================================================================

def analyze_bias(dataframe, group_column, label):
    """
    Calculates statistical bias and then uses Gemini AI to interpret 
    why this bias might exist and how to fix it.
    """
    print(f"\n  --- AI-Enhanced Bias Analysis: {label} ---")

    # 1. Standard Statistical Calculations
    group_stats = dataframe.groupby(group_column)["Approved"].agg(
        Total="count",
        Approved="sum"
    ).reset_index()
    group_stats["Approval Rate (%)"] = (group_stats["Approved"] / group_stats["Total"] * 100).round(2)
    group_stats = group_stats.sort_values("Approval Rate (%)", ascending=False)

    max_rate = group_stats["Approval Rate (%)"].max()
    min_rate = group_stats["Approval Rate (%)"].min()
    disparate_impact = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

    contingency = pd.crosstab(dataframe[group_column], dataframe["Approved"])
    if contingency.shape[0] < 2:
        chi2, p_value = 0.0, 1.0
    else:
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    bias_flag = "⚠️  BIAS DETECTED" if disparate_impact < 0.80 else "✅  FAIR"

    # 2. Gemini AI Qualitative Interpretation
    # We send the raw results to Gemini to get a human-like explanation
    stats_summary = group_stats.to_string(index=False)
    
    prompt = f"""
    Analyze these loan approval statistics for the group '{label}':
    {stats_summary}
    
    Metrics:
    - Disparate Impact Ratio: {disparate_impact} (Fairness threshold is 0.80)
    - Statistical P-Value: {p_value:.4f}
    
    Task:
    Provide a concise explanation (2 sentences) of why this specific group might be facing bias 
    (e.g., socioeconomic factors) and one specific recommendation to mitigate it.
    Return the response in this exact JSON format:
    {{"explanation": "...", "recommendation": "..."}}
    """

    try:
        response = model.generate_content(prompt)
        # Clean the response text to ensure it's valid JSON
        json_data = response.text.strip().replace('```json', '').replace('```', '')
        ai_insights = json.loads(json_data)
        explanation = ai_insights.get("explanation")
        recommendation = ai_insights.get("recommendation")
    except Exception as e:
        explanation = "AI interpretation unavailable."
        recommendation = "Manual review required."

    # Printing Results
    print(f"  Result: {bias_flag}")
    print(f"  AI Explanation: {explanation}")

    # Store results for the report
    group_stats["Disparate Impact"] = disparate_impact
    group_stats["Chi2 p-value"] = round(p_value, 4)
    group_stats["Bias Flag"] = bias_flag.replace("⚠️  ", "").replace("✅  ", "")
    group_stats["AI_Explanation"] = explanation
    group_stats["AI_Recommendation"] = recommendation

    return group_stats

# =============================================================================
# STEP 7: RUN BIAS ANALYSIS ON PROXY GROUPS
# =============================================================================

print("[ Step 7 ] Running AI-Powered analysis on proxy groups...")

# Analysis results will now include AI insights
home_bias = analyze_bias(df, "Home Ownership", "Home Ownership")
purpose_bias = analyze_bias(df, "Purpose", "Loan Purpose")
job_bias = analyze_bias(df, "Years in current job", "Years in Current Job")
term_bias = analyze_bias(df, "Term", "Loan Term")


# =============================================================================
# STEP 8: SAVE BIAS ANALYSIS RESULTS TO A SPREADSHEET
# =============================================================================

print("\n\n[ Step 8 ] Saving bias results to Excel spreadsheet...")

with pd.ExcelWriter(f"{OUTPUT_FOLDER}/3_bias_analysis_results.xlsx", engine="openpyxl") as writer:
    home_bias.to_excel(writer,    sheet_name="Home Ownership Bias",  index=False)
    purpose_bias.to_excel(writer, sheet_name="Loan Purpose Bias",    index=False)
    job_bias.to_excel(writer,     sheet_name="Employment Bias",       index=False)
    term_bias.to_excel(writer,    sheet_name="Loan Term Bias",        index=False)

print(f"  Saved: {OUTPUT_FOLDER}/3_bias_analysis_results.xlsx\n")


# =============================================================================
# STEP 9: VISUALISE APPROVAL RATES BY GROUP (Bar Charts)
# =============================================================================

print("[ Step 9 ] Plotting approval rate charts...")

def plot_approval_rates(group_stats, group_column, label, filename, threshold=0.80):
    """Creates a colour-coded bar chart for approval rates."""
    fig, ax = plt.subplots(figsize=(10, 5))

    max_rate = group_stats["Approval Rate (%)"].max()
    colors = [
        "#C44E52" if (row / max_rate) < threshold else "#55A868"
        for row in group_stats["Approval Rate (%)"]
    ]

    bars = ax.bar(group_stats[group_column], group_stats["Approval Rate (%)"],
                  color=colors, edgecolor="white", width=0.6)

    # Add value labels on top of each bar
    for bar, val in zip(bars, group_stats["Approval Rate (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Legend
    green_patch = mpatches.Patch(color="#55A868", label="Within fair range")
    red_patch   = mpatches.Patch(color="#C44E52", label="⚠️ Below 80% of highest group (potential bias)")
    ax.legend(handles=[green_patch, red_patch], loc="lower right")

    ax.set_title(f"Loan Approval Rate by {label}", fontsize=14, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Approval Rate (%)")
    ax.set_ylim(0, max_rate * 1.2)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_FOLDER}/{filename}")

plot_approval_rates(home_bias,    "Home Ownership",        "Home Ownership",      "4a_bias_home_ownership.png")
plot_approval_rates(purpose_bias, "Purpose",               "Loan Purpose",        "4b_bias_loan_purpose.png")
plot_approval_rates(job_bias,     "Years in current job",  "Years in Current Job","4c_bias_employment.png")
plot_approval_rates(term_bias,    "Term",                  "Loan Term",           "4d_bias_loan_term.png")


# =============================================================================
# STEP 10: GENERATE FINAL SUMMARY REPORT (text file)
# =============================================================================

print("\n[ Step 10 ] Generating final summary report...")

all_results = [
    ("Home Ownership",    home_bias,    "Home Ownership"),
    ("Loan Purpose",      purpose_bias, "Purpose"),
    ("Employment Length", job_bias,     "Years in current job"),
    ("Loan Term",         term_bias,    "Term"),
]

with open(f"{OUTPUT_FOLDER}/5_bias_summary_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  LOAN BIAS DETECTION — FINAL SUMMARY REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  Dataset     : {FILE_PATH}\n")
    f.write(f"  Total rows  : {len(df):,}\n")
    f.write(f"  Approved    : {approved:,} ({approved/total*100:.1f}%)\n")
    f.write(f"  Rejected    : {rejected:,} ({rejected/total*100:.1f}%)\n")
    f.write(f"  Approval rule: Credit Score >= {APPROVAL_THRESHOLD}\n\n")
    f.write("-" * 60 + "\n\n")

    for label, result_df, group_col in all_results:
        f.write(f"GROUP: {label}\n")
        f.write("-" * 40 + "\n")
        di   = result_df["Disparate Impact"].iloc[0]
        pval = result_df["Chi2 p-value"].iloc[0]
        flag = result_df["Bias Flag"].iloc[0]

        for _, row in result_df.iterrows():
            f.write(f"  {row[group_col]:<30} Approval Rate: {row['Approval Rate (%)']:>5.1f}%\n")

        f.write(f"\n  Disparate Impact : {di:.4f}  ({'⚠️  BIAS DETECTED — below 0.80 threshold' if di < 0.80 else '✅  FAIR'})\n")
        f.write(f"  Chi2 p-value     : {pval:.4f}  ({'Statistically significant' if pval < 0.05 else 'Not significant'})\n")
        f.write(f"  Verdict          : {flag}\n\n")

    f.write("=" * 60 + "\n")
    f.write("WHAT TO DO NEXT\n")
    f.write("=" * 60 + "\n\n")
    f.write("1. INVESTIGATE flagged groups (marked BIAS DETECTED).\n")
    f.write("   Ask: Is the disparity due to a legitimate financial factor,\n")
    f.write("   or is it a proxy for a protected characteristic (race, gender)?\n\n")
    f.write("2. COLLECT protected attribute data (gender, race, age) if\n")
    f.write("   available, and re-run this analysis on those columns.\n\n")
    f.write("3. CONSIDER alternative approval rules that reduce disparity\n")
    f.write("   (e.g. combine Credit Score + Income + Debt-to-Income ratio).\n\n")
    f.write("4. RE-TEST after any model changes to confirm bias was reduced.\n\n")

print(f"  Saved: {OUTPUT_FOLDER}/5_bias_summary_report.txt\n")


# =============================================================================
# DONE
# =============================================================================

print("=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"\n  Open the '{OUTPUT_FOLDER}/' folder to find:\n")
print("   1_data_overview.txt            — Dataset summary")
print("   2_data_distributions.png       — Key distribution charts")
print("   3_bias_analysis_results.xlsx   — Full bias numbers (Excel)")
print("   4a_bias_home_ownership.png     — Approval rates by home ownership")
print("   4b_bias_loan_purpose.png       — Approval rates by loan purpose")
print("   4c_bias_employment.png         — Approval rates by employment")
print("   4d_bias_loan_term.png          — Approval rates by loan term")
print("   5_bias_summary_report.txt      — Final bias verdict + next steps")
print()