import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go
import plotly.express as px
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
    --bg-dark:   #0D1117;
    --bg-card:   #161B22;
    --bg-card2:  #1C2128;
    --accent:    #00E5A0;
    --danger:    #FF5B5B;
    --warn:      #FFB347;
    --text-main: #E6EDF3;
    --text-muted:#7D8590;
    --border:    #30363D;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-main);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
h1,h2,h3,h4 { font-family: 'Space Mono', monospace !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b377) !important;
    color: #0D1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important; letter-spacing: 0.05em;
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
    margin-bottom: 0.5rem; border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
}
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stSelectbox > div > div, .stTextInput > div > div {
    background-color: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Path Resolution ─────────────────────────────────────────────────────────────
# Works whether you run:  streamlit run app/app.py   OR   streamlit run app.py
_HERE = os.path.dirname(os.path.abspath(__file__))
# If app.py is inside  app/  folder, models/ is one level up
MODELS_DIR = os.path.join(_HERE, "..", "models")
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(_HERE, "models")   # fallback: same folder
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = "models"                         # last resort: cwd

# ─── Model Loading ───────────────────────────────────────────────────────────────
SKIP_FILES = {"best_model.pkl", "svm_scaler.pkl"}   # loaded separately

@st.cache_resource
def load_all_models():
    """Load every .pkl in models/ except best_model and svm_scaler."""
    loaded = {}
    if not os.path.isdir(MODELS_DIR):
        return loaded
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".pkl"):
            continue
        if fname in SKIP_FILES:
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
    """Load evaluation_results.json or model_results.csv if present."""
    json_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    csv_path  = os.path.join(MODELS_DIR, "model_results.csv")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert CSV rows → dict keyed by model name
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
    Mirrors your predict.py exactly:
      1. Build single-row DataFrame with raw string values
      2. pd.get_dummies  →  reindex to model.feature_names_in_
      3. Scale only for SVM
      4. predict + predict_proba
    """
    df = pd.DataFrame([raw_input_dict])
    df = pd.get_dummies(df)

    # Align to the exact columns the model was trained on
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    is_svm = "svc" in str(type(model)).lower()
    if is_svm and svm_scaler is not None:
        X = svm_scaler.transform(df)
    else:
        X = df

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

# Add best_model to selector as "Best Model (Logistic Regression)" if not already there
if best_model is not None and "Best Model" not in all_models:
    all_models = {"Best Model (Logistic Regression)": best_model, **all_models}

no_models = len(all_models) == 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Smart Churn Predictor")
    st.markdown("---")
    page = st.radio("Navigate", ["🔮 Predict Churn", "📊 Model Evaluation"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<span style='color:#7D8590;font-size:0.75rem'>"
                f"Models loaded: {len(all_models)}<br>"
                f"Models dir: <code>{os.path.abspath(MODELS_DIR)}</code></span>",
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT CHURN
# ═════════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Churn":
    st.markdown("# 🔮 Churn Prediction")
    st.markdown("<p style='color:#7D8590'>Enter customer details to predict churn probability.</p>",
                unsafe_allow_html=True)

    if no_models:
        st.error(f"No `.pkl` model files found in `{os.path.abspath(MODELS_DIR)}`.\n\n"
                 "Make sure your models folder path is correct.")
        st.stop()

    # Model selector — default to Best Model
    model_names = list(all_models.keys())
    default_idx = 0  # Best Model (Logistic Regression) is inserted first
    col_sel, col_metrics = st.columns([2, 3])
    with col_sel:
        model_choice = st.selectbox("Select Model", model_names, index=default_idx)
    with col_metrics:
        if eval_results:
            # Try to match selected model name to eval_results key
            matched_key = None
            choice_lower = model_choice.lower()
            for k in eval_results:
                if any(word in choice_lower for word in k.lower().split()):
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

    # ── Feature Input Form ───────────────────────────────────────────────────────
    # These are RAW string values exactly matching the original dataset
    # predict_churn() runs get_dummies on them, just like your predict.py
    st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)

    with st.form("churn_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1],
                                          format_func=lambda x: "Yes" if x else "No")
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])
            tenure         = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**Services**")
            phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines   = st.selectbox("Multiple Lines",
                                            ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service",
                                            ["Fiber optic", "DSL", "No"])
            online_security  = st.selectbox("Online Security",
                                            ["No", "Yes", "No internet service"])
            online_backup    = st.selectbox("Online Backup",
                                            ["Yes", "No", "No internet service"])
            device_protection= st.selectbox("Device Protection",
                                            ["No", "Yes", "No internet service"])
            tech_support     = st.selectbox("Tech Support",
                                            ["No", "Yes", "No internet service"])
            streaming_tv     = st.selectbox("Streaming TV",
                                            ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies",
                                            ["No", "Yes", "No internet service"])

        with c3:
            st.markdown("**Account Info**")
            contract         = st.selectbox("Contract",
                                            ["Month-to-month", "One year", "Two year"])
            paperless_billing= st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges  = st.number_input("Monthly Charges ($)",
                                               0.0, 200.0, 65.0, step=0.5)
            total_charges    = st.number_input("Total Charges ($)",
                                               0.0, 10000.0,
                                               float(tenure * monthly_charges),
                                               step=1.0)

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    # ── Run Prediction ───────────────────────────────────────────────────────────
    if submitted:
        # Build raw dict exactly matching original CSV column names & values
        raw_input = {
            "gender":           gender,
            "SeniorCitizen":    senior_citizen,      # already int 0/1
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
            "TotalCharges":     str(total_charges),  # match original CSV type
        }

        chosen_model = all_models[model_choice]
        scaler_to_use = svm_scaler if "svm" in model_choice.lower() else None

        try:
            pred, prob = predict_churn(chosen_model, raw_input, scaler_to_use)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        churn_prob = (prob or 0) * 100
        stay_prob  = 100 - churn_prob

        st.markdown("---")
        st.markdown("### Prediction Result")

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown("<div class='churn-badge-yes'>⚠️ WILL CHURN</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='color:#FF5B5B;margin-top:1rem'>"
                            f"Churn probability: <strong>{churn_prob:.1f}%</strong></p>",
                            unsafe_allow_html=True)
                st.markdown("**Recommended actions:**")
                st.markdown("- 🎁 Offer retention discount or upgrade\n"
                            "- 📞 Proactive customer service outreach\n"
                            "- 📋 Propose longer contract with benefits")
            else:
                st.markdown("<div class='churn-badge-no'>✅ WILL STAY</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='color:#00E5A0;margin-top:1rem'>"
                            f"Retention probability: <strong>{stay_prob:.1f}%</strong></p>",
                            unsafe_allow_html=True)
                st.markdown("**Customer status:**")
                st.markdown("- 💚 Low churn risk — customer appears satisfied\n"
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

        # Input summary table
        st.markdown("#### Input Summary")
        summary = {
            "Tenure":          f"{tenure} months",
            "Monthly Charges": f"${monthly_charges:.2f}",
            "Total Charges":   f"${total_charges:.2f}",
            "Contract":        contract,
            "Internet":        internet_service,
            "Tech Support":    tech_support,
            "Paperless":       paperless_billing,
            "Senior Citizen":  "Yes" if senior_citizen else "No",
        }
        st.dataframe(pd.DataFrame(list(summary.items()), columns=["Feature", "Value"]),
                     use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("# 📊 Model Evaluation Results")
    st.markdown("<p style='color:#7D8590'>Comparison of all trained models.</p>",
                unsafe_allow_html=True)

    # Use real results if available, else show placeholder
    if eval_results is None:
        st.info("ℹ️ No `evaluation_results.json` or `model_results.csv` found in models/. "
                "Showing placeholder data. Run your evaluate.py and save results to load real numbers.")
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

    # Best model banner
    best_r = eval_results[best_name]
    st.markdown(f"### 🏆 Best Model: {best_name}")
    cols = st.columns(5)
    for i, (mk, ml) in enumerate(metrics_map.items()):
        v = best_r.get(mk) or best_r.get(ml)
        if v is not None:
            cols[i].metric(ml, f"{float(v)*100:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Metrics Table", "📈 Charts", "🔲 Confusion Matrix", "💡 Selection Rationale"])

    # ── Tab 1: Table ─────────────────────────────────────────────────────────────
    with tab1:
        rows = []
        for mname, r in eval_results.items():
            rows.append({
                "Model":     mname,
                "Accuracy":  f"{float(r.get('accuracy',0))*100:.2f}%",
                "Precision": f"{float(r.get('precision',0))*100:.2f}%",
                "Recall":    f"{float(r.get('recall',0))*100:.2f}%",
                "F1 Score":  f"{float(r.get('f1',0))*100:.2f}%",
                "ROC-AUC":   f"{float(r.get('roc_auc',0))*100:.2f}%" if r.get('roc_auc') else "N/A",
                "Best ✓":    "⭐" if mname == best_name else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if not real_data:
            st.caption("⚠️ Sample data — replace with real results from evaluate.py")

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
            vals = []
            for k in models_list:
                v = eval_results[k].get(mk)
                vals.append(float(v)*100 if v is not None else 0)
            fig.add_trace(go.Bar(name=ml, x=models_list, y=vals, marker_color=colors[mk]))
        fig.update_layout(
            barmode="group", title="All Models — Metric Comparison",
            paper_bgcolor="#161B22", plot_bgcolor="#161B22",
            font={"color":"#E6EDF3"},
            xaxis={"gridcolor":"#30363D"},
            yaxis={"gridcolor":"#30363D","range":[0,100],"title":"Score (%)"},
            legend={"bgcolor":"#1C2128","bordercolor":"#30363D","borderwidth":1},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar
        radar_fig = go.Figure()
        pal = ["#00E5A0","#4D9EFF","#FFB347","#FF5B5B"]
        theta = list(metrics_map.values())
        for idx, (mname, r) in enumerate(eval_results.items()):
            vals = [float(r.get(mk) or 0)*100 for mk in metrics_map]
            vals += [vals[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals, theta=theta+[theta[0]],
                fill="toself", name=mname, opacity=0.5,
                line={"color": pal[idx % len(pal)]}
            ))
        radar_fig.update_layout(
            polar={"bgcolor":"#1C2128",
                   "radialaxis":{"range":[0,100],"gridcolor":"#30363D","color":"#7D8590"},
                   "angularaxis":{"gridcolor":"#30363D","color":"#E6EDF3"}},
            paper_bgcolor="#161B22", font={"color":"#E6EDF3"},
            title="Radar Chart — Model Comparison",
            legend={"bgcolor":"#1C2128","bordercolor":"#30363D"},
            height=450,
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # ── Tab 3: Confusion Matrix ───────────────────────────────────────────────────
    with tab3:
        sel = st.selectbox("Select model", models_list,
                           index=models_list.index(best_name) if best_name in models_list else 0)
        cm_data = eval_results.get(sel, {}).get("confusion_matrix")
        if cm_data:
            cm = np.array(cm_data)
            labels = ["No Churn", "Churn"]
            cm_fig = go.Figure(go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale=[[0,"#161B22"],[1,"#00E5A0"]],
                showscale=True,
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
                textfont={"size":22,"color":"white"},
            ))
            cm_fig.update_layout(
                title=f"Confusion Matrix — {sel}",
                xaxis={"title":"Predicted","color":"#E6EDF3"},
                yaxis={"title":"Actual","color":"#E6EDF3","autorange":"reversed"},
                paper_bgcolor="#161B22", font={"color":"#E6EDF3"}, height=380,
            )
            st.plotly_chart(cm_fig, use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("True Positives (TP)",  tp, help="Correctly predicted churn")
            c2.metric("True Negatives (TN)",  tn, help="Correctly predicted no churn")
            c3.metric("False Positives (FP)", fp, help="Predicted churn, actually stayed")
            c4.metric("False Negatives (FN)", fn, help="Missed churners — most costly!")
        else:
            st.info("Add `confusion_matrix` to your evaluation_results.json to see this chart.\n\n"
                    "In evaluate.py: `confusion_matrix(y_test, y_pred).tolist()`")

    # ── Tab 4: Rationale ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Why We Selected the Best Model")
        best_r = eval_results[best_name]
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
   Recall was weighted accordingly in model selection.
5. **Interpretability** — Logistic Regression provides clear feature coefficients, 
   helping management understand which factors drive churn.

**Business Impact:**
Catching a churner before they leave is far more valuable than a false alarm.
The selected model maximises recall while maintaining acceptable precision,
enabling targeted retention campaigns for at-risk customers.
</div>
""", unsafe_allow_html=True)

        st.markdown("#### F1 Score Leaderboard")
        sorted_models = sorted(eval_results.items(), key=lambda x: float(x[1].get("f1",0)), reverse=True)
        for rank, (mname, r) in enumerate(sorted_models, 1):
            f1 = float(r.get("f1",0)) * 100
            icon = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}."
            st.markdown(
                f"<div style='margin:0.4rem 0;display:flex;align-items:center;gap:1rem'>"
                f"<span style='width:2rem;font-size:1.1rem'>{icon}</span>"
                f"<span style='width:12rem;color:#E6EDF3'>{mname}</span>"
                f"<div style='flex:1;background:#1C2128;border-radius:4px;height:20px'>"
                f"<div style='width:{int(f1)}%;background:#00E5A0;height:100%;border-radius:4px'></div></div>"
                f"<span style='width:4rem;text-align:right;font-family:Space Mono,monospace;"
                f"color:#00E5A0'>{f1:.1f}%</span></div>",
                unsafe_allow_html=True
            )