import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg-dark:   #0D1117; --bg-card:  #161B22; --bg-card2: #1C2128;
    --accent:    #00E5A0; --danger:   #FF5B5B; --warn:     #FFB347;
    --text-main: #E6EDF3; --text-muted:#7D8590; --border:  #30363D;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-main); font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
h1,h2,h3,h4 { font-family: 'Space Mono', monospace !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b377) !important;
    color: #0D1117 !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700; border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; font-size: 0.9rem !important;
    letter-spacing: 0.05em; transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}
.churn-badge-yes {
    background: rgba(255,91,91,0.15); border: 1.5px solid var(--danger);
    color: var(--danger); border-radius: 50px; padding: 0.4rem 1.5rem;
    font-family: 'Space Mono', monospace; font-weight: 700;
    font-size: 1.1rem; display: inline-block;
}
.churn-badge-no {
    background: rgba(0,229,160,0.12); border: 1.5px solid var(--accent);
    color: var(--accent); border-radius: 50px; padding: 0.4rem 1.5rem;
    font-family: 'Space Mono', monospace; font-weight: 700;
    font-size: 1.1rem; display: inline-block;
}
.section-header {
    font-family: 'Space Mono', monospace; color: var(--accent);
    font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 0.75rem; border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
}
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stSelectbox > div > div, .stTextInput > div > div {
    background-color: var(--bg-card2) !important;
    border-color: var(--border) !important; color: var(--text-main) !important;
}
.all-model-card {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
}
.all-model-card.churn {
    border-left: 4px solid var(--danger);
}
.all-model-card.stay {
    border-left: 4px solid var(--accent);
}
/* Radio button styling for green color */
[data-testid="stRadio"] input[type="radio"] {
    accent-color: var(--accent) !important;
    width: 1.2rem !important;
    height: 1.2rem !important;
}
[data-testid="stRadio"] input[type="radio"]:checked {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="stRadio"] label {
    color: var(--text-main) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Path Resolution ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "..", "models")
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(_HERE, "models")
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = "models"

# ─── Model Loading ───────────────────────────────────────────────────────────────
SKIP_FILES = {"best_model.pkl", "svm_scaler.pkl", "logistic_regression.pkl"}

@st.cache_resource
def load_all_models():
    loaded = {}
    if not os.path.isdir(MODELS_DIR):
        return loaded
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".pkl") or fname in SKIP_FILES:
            continue
        name = fname.replace(".pkl", "").replace("_", " ").title()
        try:
            loaded[name] = joblib.load(os.path.join(MODELS_DIR, fname))
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")
    return loaded

@st.cache_resource
def load_best_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_svm_scaler():
    path = os.path.join(MODELS_DIR, "svm_scaler.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_eval_results():
    json_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    csv_path  = os.path.join(MODELS_DIR, "model_results.csv")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            key = str(row.get("Model", "")).replace("_", " ").title()
            result[key] = {
                "accuracy":  row.get("Accuracy",  row.get("accuracy",  0)),
                "precision": row.get("Precision", row.get("precision", 0)),
                "recall":    row.get("Recall",    row.get("recall",    0)),
                "f1":        row.get("F1-Score",  row.get("f1",        0)),
                "roc_auc":   row.get("ROC-AUC",   row.get("roc_auc",   None)),
            }
        return result
    return None

# ─── Prediction Helper ───────────────────────────────────────────────────────────
def predict_churn(model, raw_input_dict, svm_scaler=None):
    """
    Encode features with drop_first=True to match training preprocessing.
    This ensures SVM and all models work with the same feature set.
    """
    # Define categorical columns with their possible values (in order)
    # NOTE: We'll drop the FIRST value (reference category) like drop_first=True does
    categorical_cols = {
        "gender": ["Female", "Male"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No internet service", "Yes"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    }
    
    # Create DataFrame
    df = pd.DataFrame([raw_input_dict])
    dummies_df = pd.DataFrame()
    
    # Add numeric columns as-is (convert to float to handle string inputs)
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df.columns:
            dummies_df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create dummy columns with drop_first logic
    # (skip the first category of each categorical variable)
    for col, possible_values in categorical_cols.items():
        if col in df.columns:
            # Skip first value (reference category), create dummies for rest
            for val in possible_values[1:]:  # [1:] to drop_first
                col_name = f"{col}_{val}"
                dummies_df[col_name] = (df[col].values == val).astype(int)
    
    # Align to model's expected features
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in dummies_df.columns:
                dummies_df[col] = 0
        dummies_df = dummies_df[expected_cols]
    
    # Scale for SVM if needed
    is_svm = "svc" in str(type(model)).lower()
    X = svm_scaler.transform(dummies_df) if (is_svm and svm_scaler is not None) else dummies_df
    
    prediction = int(model.predict(X)[0])
    
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probability = float(1 / (1 + np.exp(-scores[0])))
    
    return prediction, probability

# ─── Load everything ─────────────────────────────────────────────────────────────
all_models   = load_all_models()
best_model   = load_best_model()
svm_scaler   = load_svm_scaler()
eval_results = load_eval_results()

# Determine best model name dynamically from eval_results (based on highest F1 score)
best_model_name = "Logistic Regression"  # fallback
if eval_results:
    best_model_name = max(eval_results.items(), 
                         key=lambda x: float(x[1].get("f1", 0)))[0]

if best_model is not None and "Best Model" not in all_models:
    all_models = {f"Best Model ({best_model_name})": best_model, **all_models}

no_models = len(all_models) == 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Smart Churn Predictor")
    st.markdown("---")
    page = st.radio("Navigate", ["Predict Churn", " Model Evaluation"],
                    label_visibility="collapsed")
    st.markdown("---")
    # CHANGE 2: removed models dir path line — only show count
    st.markdown(f"<span style='color:#7D8590;font-size:0.75rem'>"
                f"Models loaded: {len(all_models)}</span>",
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT CHURN
# ═════════════════════════════════════════════════════════════════════════════════
if page == "Predict Churn":
    st.markdown("# Churn Prediction")
    st.markdown("<p style='color:#7D8590'>Enter customer details to predict churn probability.</p>",
                unsafe_allow_html=True)

    if no_models:
        st.error("No `.pkl` model files found. Make sure your models folder is correct.")
        st.stop()

    # Model selector
    model_names = list(all_models.keys())
    col_sel, col_metrics = st.columns([2, 3])
    with col_sel:
        model_choice = st.selectbox("Select Model", model_names, index=0)
    with col_metrics:
        if eval_results:
            matched_key = None
            for k in eval_results:
                if any(word in model_choice.lower() for word in k.lower().split()):
                    matched_key = k
                    break
            if matched_key:
                r = eval_results[matched_key]
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{float(r.get('accuracy',0))*100:.1f}%")
                c2.metric("F1 Score", f"{float(r.get('f1',0))*100:.1f}%")
                roc = r.get("roc_auc") or r.get("ROC-AUC")
                c3.metric("ROC-AUC",  f"{float(roc)*100:.1f}%" if roc else "N/A")

    st.markdown("---")

    # ── CHANGE 4: Row-wise layout — 4 rows of fields so no column is overloaded ──
    st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)

    with st.form("churn_form"):

        # Row 1 — Demographics (5 fields across 5 columns)
        st.markdown("**👤 Demographics**")
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        gender         = r1c1.selectbox("Gender", ["Male", "Female"])
        senior_citizen = r1c2.selectbox("Senior Citizen", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
        partner        = r1c3.selectbox("Partner", ["Yes", "No"])
        dependents     = r1c4.selectbox("Dependents", ["Yes", "No"])
        tenure         = r1c5.slider("Tenure (months)", 0, 72, 12)

        st.markdown("---")

        # Row 2 — Phone & Internet (4 fields)
        st.markdown("**📞 Phone & Internet**")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        phone_service    = r2c1.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines   = r2c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = r2c3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security  = r2c4.selectbox("Online Security", ["No", "Yes", "No internet service"])

        st.markdown("---")

        # Row 3 — Online Services (5 fields)
        st.markdown("**🌐 Online Services**")
        r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
        online_backup     = r3c1.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_protection = r3c2.selectbox("Device Protection",["No",  "Yes","No internet service"])
        tech_support      = r3c3.selectbox("Tech Support",     ["No",  "Yes","No internet service"])
        streaming_tv      = r3c4.selectbox("Streaming TV",     ["No",  "Yes","No internet service"])
        streaming_movies  = r3c5.selectbox("Streaming Movies", ["No",  "Yes","No internet service"])

        st.markdown("---")

        # Row 4 — Account Info (5 fields)
        st.markdown("**💳 Account Info**")
        r4c1, r4c2, r4c3, r4c4, r4c5 = st.columns(5)
        contract          = r4c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = r4c2.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method    = r4c3.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges   = r4c4.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges     = r4c5.number_input("Total Charges ($)",   0.0, 10000.0,
                                              float(tenure * monthly_charges), step=1.0)

        st.markdown("---")

        # Two submit buttons side by side
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submitted = st.form_submit_button("🔍 Predict Churn Using Selected Model", use_container_width=True)
        with btn_col2:
            submitted_all = st.form_submit_button("🤖 Use All Models", use_container_width=True)

    # ── Build raw input dict (shared by both buttons) ────────────────────────────
    raw_input = {
        "gender":           gender,
        "SeniorCitizen":    senior_citizen,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     str(total_charges),
    }

    # ── SINGLE MODEL PREDICTION ──────────────────────────────────────────────────
    if submitted:
        chosen_model  = all_models[model_choice]
        scaler_to_use = svm_scaler if "svm" in model_choice.lower() else None
        try:
            pred, prob = predict_churn(chosen_model, raw_input, scaler_to_use)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        churn_prob = (prob or 0) * 100
        stay_prob  = 100 - churn_prob

        st.markdown("---")
        st.markdown(f"### Result — {model_choice}")
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown("<div class='churn-badge-yes'>⚠️ WILL CHURN</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='color:#FF5B5B;margin-top:1rem'>"
                            f"Churn probability: <strong>{churn_prob:.1f}%</strong></p>",
                            unsafe_allow_html=True)
                st.markdown("**Recommended actions:**\n"
                            "- 🎁 Offer retention discount or upgrade\n"
                            "- 📞 Proactive customer service outreach\n"
                            "- 📋 Propose longer contract with benefits")
            else:
                st.markdown("<div class='churn-badge-no'>✅ WILL STAY</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='color:#00E5A0;margin-top:1rem'>"
                            f"Retention probability: <strong>{stay_prob:.1f}%</strong></p>",
                            unsafe_allow_html=True)
                st.markdown("**Customer status:**\n"
                            "- 💚 Low churn risk — customer appears satisfied\n"
                            "- 📈 Consider upsell opportunities\n"
                            "- 🌟 Loyalty program candidate")
            st.markdown("</div>", unsafe_allow_html=True)

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                title={"text": "Churn Probability (%)", "font": {"color": "#E6EDF3"}},
                number={"suffix": "%", "font": {"color": "#E6EDF3", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7D8590"},
                    "bar":  {"color": "#FF5B5B" if pred == 1 else "#00E5A0"},
                    "bgcolor": "#1C2128", "bordercolor": "#30363D",
                    "steps": [
                        {"range": [0,  30], "color": "rgba(0,229,160,0.1)"},
                        {"range": [30, 60], "color": "rgba(255,179,71,0.1)"},
                        {"range": [60,100], "color": "rgba(255,91,91,0.1)"},
                    ],
                    "threshold": {"line": {"color": "#FFB347", "width": 3}, "value": 50}
                }
            ))
            fig.update_layout(
                paper_bgcolor="#161B22", plot_bgcolor="#161B22",
                font={"color": "#E6EDF3"}, height=280,
                margin=dict(t=40, b=20, l=30, r=30)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── CHANGE 5: ALL MODELS PREDICTION ─────────────────────────────────────────
    if submitted_all:
        st.markdown("---")
        st.markdown("### 🤖 All Models Comparison")
        st.markdown("<p style='color:#7D8590'>Prediction from every trained model for this customer.</p>",
                    unsafe_allow_html=True)

        all_results = []
        for mname, mobj in all_models.items():
            scaler_to_use = svm_scaler if "svm" in mname.lower() else None
            try:
                p, prob = predict_churn(mobj, raw_input, scaler_to_use)
                all_results.append({"name": mname, "pred": p, "prob": (prob or 0) * 100})
            except Exception as e:
                all_results.append({"name": mname, "pred": -1, "prob": 0, "error": str(e)})

        # Summary counts
        churn_count = sum(1 for r in all_results if r["pred"] == 1)
        stay_count  = sum(1 for r in all_results if r["pred"] == 0)
        total_count = len(all_results)

        verdict_col, bar_col = st.columns([1, 2])
        with verdict_col:
            st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
            if churn_count > stay_count:
                st.markdown("<div class='churn-badge-yes' style='margin-bottom:0.75rem'>⚠️ MAJORITY: CHURN</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='churn-badge-no' style='margin-bottom:0.75rem'>✅ MAJORITY: STAY</div>",
                            unsafe_allow_html=True)
            st.markdown(f"<p style='margin-top:0.75rem;color:#E6EDF3'>"
                        f"<strong style='color:#FF5B5B'>{churn_count}</strong> model(s) predict churn<br>"
                        f"<strong style='color:#00E5A0'>{stay_count}</strong> model(s) predict stay</p>",
                        unsafe_allow_html=True)
            avg_prob = np.mean([r["prob"] for r in all_results if r["pred"] >= 0])
            st.markdown(f"<p style='color:#7D8590;font-size:0.85rem'>Avg churn probability<br>"
                        f"<span style='font-family:Space Mono,monospace;font-size:1.4rem;"
                        f"color:#FFB347'>{avg_prob:.1f}%</span></p>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with bar_col:
            # Horizontal bar chart of churn probabilities
            sorted_res = sorted(all_results, key=lambda x: x["prob"], reverse=True)
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                y=[r["name"] for r in sorted_res],
                x=[r["prob"] for r in sorted_res],
                orientation="h",
                marker_color=["#FF5B5B" if r["pred"] == 1 else "#00E5A0" for r in sorted_res],
                text=[f"{r['prob']:.1f}%" for r in sorted_res],
                textposition="outside",
            ))
            bar_fig.add_vline(x=50, line_dash="dash", line_color="#FFB347",
                              annotation_text="50% threshold", annotation_font_color="#FFB347")
            bar_fig.update_layout(
                title="Churn Probability by Model",
                xaxis={"range": [0, 115], "title": "Churn Probability (%)",
                       "gridcolor": "#30363D"},
                yaxis={"gridcolor": "#30363D"},
                paper_bgcolor="#161B22", plot_bgcolor="#161B22",
                font={"color": "#E6EDF3"}, height=60 + 60 * len(all_results),
                margin=dict(t=40, b=40, l=160, r=60),
                showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # Detailed cards — one per model
        st.markdown("#### Per-Model Detail")
        cols_per_row = 2
        model_items  = list(all_results)
        for i in range(0, len(model_items), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(row_cols):
                idx = i + j
                if idx >= len(model_items):
                    break
                r = model_items[idx]
                if r.get("error"):
                    col.error(f"**{r['name']}** — Error: {r['error']}")
                    continue
                verdict    = "⚠️ WILL CHURN" if r["pred"] == 1 else "✅ WILL STAY"
                card_class = "churn" if r["pred"] == 1 else "stay"
                color      = "#FF5B5B" if r["pred"] == 1 else "#00E5A0"
                col.markdown(
                    f"<div class='all-model-card {card_class}'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-family:Space Mono,monospace;font-size:0.85rem;"
                    f"color:#E6EDF3'>{r['name']}</span>"
                    f"<span style='color:{color};font-weight:700;font-size:0.85rem'>{verdict}</span>"
                    f"</div>"
                    f"<div style='margin-top:0.6rem;background:#0D1117;border-radius:4px;height:8px'>"
                    f"<div style='width:{r['prob']:.1f}%;background:{color};"
                    f"height:100%;border-radius:4px;transition:width 0.4s'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;margin-top:0.3rem'>"
                    f"<span style='color:#7D8590;font-size:0.75rem'>Churn probability</span>"
                    f"<span style='font-family:Space Mono,monospace;color:{color};"
                    f"font-size:0.9rem'>{r['prob']:.1f}%</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )


# ═════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("# 📊 Model Evaluation Results")
    st.markdown("<p style='color:#7D8590'>Comparison of all trained models.</p>",
                unsafe_allow_html=True)

    if eval_results is None:
        st.info("ℹ️ No `evaluation_results.json` or `model_results.csv` found in models/. "
                "Showing placeholder data. Run evaluate.py to load real numbers.")
        eval_results = {
            "Logistic Regression": {"accuracy":0.801,"precision":0.659,"recall":0.543,"f1":0.595,"roc_auc":0.843,"confusion_matrix":[[947,98],[171,202]]},
            "Decision Tree":       {"accuracy":0.727,"precision":0.492,"recall":0.491,"f1":0.491,"roc_auc":0.688,"confusion_matrix":[[892,153],[234,139]]},
            "Random Forest":       {"accuracy":0.799,"precision":0.649,"recall":0.476,"f1":0.549,"roc_auc":0.827,"confusion_matrix":[[966,79],[203,170]]},
            "Svm":                 {"accuracy":0.795,"precision":0.632,"recall":0.523,"f1":0.572,"roc_auc":0.836,"confusion_matrix":[[944,101],[174,199]]},
        }
        real_data = False
    else:
        real_data = True

    models_list = list(eval_results.keys())
    best_name   = max(eval_results, key=lambda k: float(eval_results[k].get("f1", 0)))
    metrics_map = {"accuracy":"Accuracy","precision":"Precision",
                   "recall":"Recall","f1":"F1 Score","roc_auc":"ROC-AUC"}

    best_r = eval_results[best_name]
    st.markdown(f"### Best Model: {best_name}")
    cols = st.columns(5)
    for i, (mk, ml) in enumerate(metrics_map.items()):
        v = best_r.get(mk) or best_r.get(ml)
        if v is not None:
            cols[i].metric(ml, f"{float(v)*100:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Metrics Table", "📈 Charts", "📉 ROC Curves",
         "🔲 Confusion Matrix", "💡 Selection Rationale"])

    # ── Tab 1: Metrics Table ─────────────────────────────────────────────────────
    with tab1:
        rows = []
        for mname, r in eval_results.items():
            roc = r.get("roc_auc") or r.get("ROC-AUC")
            rows.append({
                "Model":     mname,
                "Accuracy":  f"{float(r.get('accuracy',0))*100:.2f}%",
                "Precision": f"{float(r.get('precision',0))*100:.2f}%",
                "Recall":    f"{float(r.get('recall',0))*100:.2f}%",
                "F1 Score":  f"{float(r.get('f1',0))*100:.2f}%",
                "ROC-AUC":   f"{float(roc)*100:.2f}%" if roc else "N/A",
                # CHANGE 3: "Best ✓" column removed entirely
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if not real_data:
            st.caption("⚠️ Sample data — run evaluate.py to replace with real results.")
        st.markdown("---")
        st.markdown("**Why F1 Score for model selection?**\n\n"
                    "The Telco churn dataset is imbalanced (~26% churn). Accuracy alone "
                    "is misleading — a naive 'never churn' model scores ~74%. F1 Score "
                    "balances **Precision** (avoiding false alarms) and **Recall** "
                    "(catching real churners), making it the correct primary metric.")

    # ── Tab 2: Charts ────────────────────────────────────────────────────────────
    with tab2:
        colors = {"accuracy":"#4D9EFF","precision":"#00E5A0",
                  "recall":"#FFB347","f1":"#FF5B5B","roc_auc":"#C792EA"}
        fig = go.Figure()
        for mk, ml in metrics_map.items():
            vals = [float(eval_results[k].get(mk) or 0)*100 for k in models_list]
            fig.add_trace(go.Bar(name=ml, x=models_list, y=vals, marker_color=colors[mk]))
        fig.update_layout(
            barmode="group", title="All Models — Metric Comparison",
            paper_bgcolor="#161B22", plot_bgcolor="#161B22", font={"color":"#E6EDF3"},
            xaxis={"gridcolor":"#30363D"},
            yaxis={"gridcolor":"#30363D","range":[0,100],"title":"Score (%)"},
            legend={"bgcolor":"#1C2128","bordercolor":"#30363D","borderwidth":1},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        radar_fig = go.Figure()
        pal = ["#00E5A0","#4D9EFF","#FFB347","#FF5B5B"]
        theta = list(metrics_map.values())
        for idx, (mname, r) in enumerate(eval_results.items()):
            vals = [float(r.get(mk) or 0)*100 for mk in metrics_map] 
            vals += [vals[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals, theta=theta+[theta[0]], fill="toself",
                name=mname, opacity=0.5, line={"color": pal[idx % len(pal)]}
            ))
        radar_fig.update_layout(
            polar={"bgcolor":"#1C2128",
                   "radialaxis":{"range":[0,100],"gridcolor":"#30363D","color":"#7D8590"},
                   "angularaxis":{"gridcolor":"#30363D","color":"#E6EDF3"}},
            paper_bgcolor="#161B22", font={"color":"#E6EDF3"},
            title="Radar Chart — Model Comparison",
            legend={"bgcolor":"#1C2128","bordercolor":"#30363D"}, height=450,
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # ── Tab 3: ROC Curves ────────────────────────────────────────────────────────
    with tab3:
        pal = ["#00E5A0","#4D9EFF","#FFB347","#FF5B5B","#C792EA"]
        has_roc_data = any(eval_results[k].get("fpr") is not None for k in models_list)

        if has_roc_data:
            st.markdown("#### Individual ROC Curves")
            n_models = len(models_list)
            for i in range(0, n_models, 2):
                row_cols = st.columns(2)
                for j, col in enumerate(row_cols):
                    idx = i + j
                    if idx >= n_models:
                        break
                    mname = models_list[idx]
                    r   = eval_results[mname]
                    fpr = r.get("fpr"); tpr = r.get("tpr"); auc = r.get("roc_auc")
                    if fpr and tpr:
                        roc_fig = go.Figure()
                        roc_fig.add_trace(go.Scatter(
                            x=[0,1], y=[0,1], mode="lines", name="Random",
                            line={"dash":"dash","color":"#7D8590","width":1}))
                        roc_fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"AUC = {auc:.3f}" if auc else mname,
                            line={"color":pal[idx%len(pal)],"width":2.5},
                            fill="tozeroy",
                            fillcolor=f"rgba({','.join(str(int(pal[idx%len(pal)].lstrip('#')[k:k+2],16)) for k in (0,2,4))},0.08)",
                        ))
                        roc_fig.update_layout(
                            title={"text":mname,"font":{"size":13}},
                            xaxis={"title":"False Positive Rate","gridcolor":"#30363D","range":[0,1]},
                            yaxis={"title":"True Positive Rate","gridcolor":"#30363D","range":[0,1]},
                            paper_bgcolor="#161B22", plot_bgcolor="#161B22",
                            font={"color":"#E6EDF3","size":11},
                            legend={"bgcolor":"#1C2128","bordercolor":"#30363D"},
                            height=320, margin=dict(t=40,b=40,l=50,r=20),
                        )
                        col.plotly_chart(roc_fig, use_container_width=True)

            st.markdown("#### Combined ROC Curve — All Models")
            combined_fig = go.Figure()
            combined_fig.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines", name="Random (AUC = 0.50)",
                line={"dash":"dash","color":"#7D8590","width":1.5}))
            for idx, mname in enumerate(models_list):
                r = eval_results[mname]; fpr = r.get("fpr"); tpr = r.get("tpr"); auc = r.get("roc_auc")
                if fpr and tpr:
                    combined_fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{mname}  (AUC = {auc:.3f})" if auc else mname,
                        line={"color":pal[idx%len(pal)],"width":2.5}))
            combined_fig.update_layout(
                title="ROC Curve Comparison — All Models",
                xaxis={"title":"False Positive Rate","gridcolor":"#30363D","range":[0,1],"zeroline":False},
                yaxis={"title":"True Positive Rate","gridcolor":"#30363D","range":[0,1],"zeroline":False},
                paper_bgcolor="#161B22", plot_bgcolor="#161B22", font={"color":"#E6EDF3"},
                legend={"bgcolor":"#1C2128","bordercolor":"#30363D","borderwidth":1}, height=480,
            )
            st.plotly_chart(combined_fig, use_container_width=True)

            st.markdown("#### AUC Score Summary")
            for rank, (mname, auc) in enumerate(
                    sorted([(k, float(eval_results[k].get("roc_auc") or 0)) for k in models_list],
                           key=lambda x: x[1], reverse=True), 1):
                # Consistent styling for all ranks - numbered badge
                st.markdown(
                    f"<div style='margin:0.4rem 0;display:flex;align-items:center;gap:1rem'>"
                    f"<div style='width:2rem;height:2rem;background:linear-gradient(135deg, var(--accent), #00b377);border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#0D1117;font-size:0.9rem'>{rank}</div>"
                    f"<span style='width:14rem;color:#E6EDF3'>{mname}</span>"
                    f"<div style='flex:1;background:#1C2128;border-radius:4px;height:20px'>"
                    f"<div style='width:{int(auc*100)}%;background:#4D9EFF;height:100%;border-radius:4px'></div></div>"
                    f"<span style='width:4rem;text-align:right;font-family:Space Mono,monospace;"
                    f"color:#4D9EFF'>{auc*100:.2f}%</span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("ROC curve data not found. Run the updated `evaluate.py` to generate it.")
            roc_img = os.path.join(MODELS_DIR, "roc_curves.png")
            if os.path.exists(roc_img):
                st.image(roc_img, caption="ROC Curves (static image from evaluate.py)")

    # ── CHANGE 1: Tab 4 — All confusion matrices side by side ───────────────────
    with tab4:
        st.markdown("#### Confusion Matrix — All Models")
        st.markdown("<p style='color:#7D8590;font-size:0.85rem'>Compare how each model classifies churners vs non-churners.</p>",
                    unsafe_allow_html=True)

        # Check if any model has confusion matrix data
        has_cm = any(eval_results[k].get("confusion_matrix") for k in models_list)

        if has_cm:
            # Show all matrices in a 2-column grid
            for i in range(0, len(models_list), 2):
                row_cols = st.columns(2)
                for j, col in enumerate(row_cols):
                    idx = i + j
                    if idx >= len(models_list):
                        break
                    mname   = models_list[idx]
                    cm_data = eval_results[mname].get("confusion_matrix")
                    if not cm_data:
                        col.info(f"No data for {mname}")
                        continue
                    cm     = np.array(cm_data)
                    labels = ["No Churn", "Churn"]
                    is_best = mname == best_name

                    cm_fig = go.Figure(go.Heatmap(
                        z=cm, x=labels, y=labels,
                        colorscale=[[0,"#161B22"],[1,"#00E5A0" if not is_best else "#4D9EFF"]],
                        showscale=False,
                        text=[[str(v) for v in row] for row in cm],
                        texttemplate="%{text}",
                        textfont={"size":24,"color":"white"},
                    ))
                    title_suffix = " ⭐" if is_best else ""
                    cm_fig.update_layout(
                        title={"text": f"{mname}{title_suffix}", "font":{"size":13}},
                        xaxis={"title":"Predicted","color":"#E6EDF3","side":"bottom"},
                        yaxis={"title":"Actual","color":"#E6EDF3","autorange":"reversed"},
                        paper_bgcolor="#161B22", font={"color":"#E6EDF3"},
                        height=320, margin=dict(t=50,b=50,l=80,r=20),
                    )
                    col.plotly_chart(cm_fig, use_container_width=True)

                    # Mini stats under each matrix
                    tn, fp, fn, tp = cm.ravel()
                    mc1, mc2, mc3, mc4 = col.columns(4)
                    mc1.metric("TP", tp)
                    mc2.metric("TN", tn)
                    mc3.metric("FP", fp)
                    mc4.metric("FN", fn)

                st.markdown("---")

            st.markdown(
                "<p style='color:#7D8590;font-size:0.8rem'>"
                "TP = True Positives (correctly caught churners) &nbsp;|&nbsp; "
                "TN = True Negatives &nbsp;|&nbsp; "
                "FP = False alarms &nbsp;|&nbsp; "
                "FN = Missed churners (most costly)"
                "</p>", unsafe_allow_html=True)
        else:
            st.info("Run the updated `evaluate.py` to generate confusion matrix data.")

    # ── Tab 5: Selection Rationale ────────────────────────────────────────────────
    with tab5:
        st.markdown("### Key Factors in Model Selection")
        best_r  = eval_results[best_name]
        roc_val = best_r.get("roc_auc")
        st.markdown(f"""
<div class='card'>
<h4 style='color:#00E5A0'>✅ {best_name} — Selected Model</h4>

**Performance (Test Set):**
- Accuracy:  **{float(best_r.get('accuracy',0))*100:.2f}%**
- Precision: **{float(best_r.get('precision',0))*100:.2f}%**
- Recall:    **{float(best_r.get('recall',0))*100:.2f}%**
- F1 Score:  **{float(best_r.get('f1',0))*100:.2f}%** ← *Primary selection metric*
- ROC-AUC:   **{float(roc_val)*100:.2f}%** ← *Threshold-independent discriminability*

**Selection Reasoning:**
1. **Highest F1 Score** — balances precision and recall on an imbalanced dataset (~26% churn).
2. **Strong ROC-AUC** — model discriminates well between churners and non-churners across all thresholds.
3. **5-Fold Cross-Validation** — consistent CV scores confirmed model generalises well (no overfitting).
4. **Business cost** — false negatives (missed churners) cost more than false positives.
5. **Interpretability** — Logistic Regression provides clear feature coefficients, helping management understand which factors drive churn.

**Business Impact:**
Catching a churner before they leave is far more valuable than a false alarm.
The selected model maximises recall while maintaining acceptable precision,
enabling targeted retention campaigns for at-risk customers.
</div>
""", unsafe_allow_html=True)

        st.markdown("#### F1 Score Leaderboard")
        for rank, (mname, r) in enumerate(
                sorted(eval_results.items(), key=lambda x: float(x[1].get("f1",0)), reverse=True), 1):
            f1   = float(r.get("f1",0)) * 100
            # Consistent styling for all ranks - numbered badge
            st.markdown(
                f"<div style='margin:0.4rem 0;display:flex;align-items:center;gap:1rem'>"
                f"<div style='width:2rem;height:2rem;background:linear-gradient(135deg, var(--accent), #00b377);border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#0D1117;font-size:0.9rem'>{rank}</div>"
                f"<span style='width:12rem;color:#E6EDF3'>{mname}</span>"
                f"<div style='flex:1;background:#1C2128;border-radius:4px;height:20px'>"
                f"<div style='width:{int(f1)}%;background:#00E5A0;height:100%;border-radius:4px'></div></div>"
                f"<span style='width:4rem;text-align:right;font-family:Space Mono,monospace;"
                f"color:#00E5A0'>{f1:.1f}%</span></div>",
                unsafe_allow_html=True)
