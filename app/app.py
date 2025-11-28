# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import io

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# -------- CONFIG: set your model filename here ----------
# Default expects models/model.pkl relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.joinpath("..", "models", "breast_cancer_rf_model.pkl").resolve()  # change if needed
# --------------------------------------------------------

st.title("🧬 Breast Cancer Prediction Model")
st.caption("Note:- Demo — not for medical use. Use for portfolio / learning only and educational pupose.")

# --------- Load model safely ---------
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}\nUpdate MODEL_PATH in app.py")
    model = joblib.load(path)
    return model

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Try to get feature names expected by the model (if available)
feature_names = None
try:
    # sklearn estimators/pipelines often expose feature_names_in_
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    # if model is a dict (artifact), try common patterns
    elif isinstance(model, dict) and "pipeline" in model:
        pipeline = model["pipeline"]
        if hasattr(pipeline, "feature_names_in_"):
            feature_names = list(pipeline.feature_names_in_)
        elif hasattr(pipeline, "named_steps"):
            # attempt to pull final estimator
            try:
                final = pipeline.named_steps[list(pipeline.named_steps)[-1]]
                if hasattr(final, "feature_names_in_"):
                    feature_names = list(final.feature_names_in_)
            except Exception:
                pass
    # also check for common dict key "features"
    elif isinstance(model, dict) and "features" in model:
        feature_names = list(model["features"])
except Exception:
    feature_names = None

st.sidebar.header("Options")
st.sidebar.write(f"Model path: `{MODEL_PATH}`")
if feature_names:
    st.sidebar.success(f"Model expects {len(feature_names)} features.")
else:
    st.sidebar.info("Model feature names not detected — app will accept CSV with proper columns or manual input (you decide).")

# ---------- Input choice ----------
mode = st.radio("Choose input mode:", ("Single record (manual)", "Batch via CSV upload"))

if mode == "Single record (manual)":
    st.subheader("Enter feature values")
    # If model exposes feature names, auto-generate inputs
    if feature_names:
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            col = cols[i % 2]
            # numeric input by default; user can paste categorical as text if needed
            input_data[feat] = [col.text_input(label=feat, value="0")]
        # Try to coerce numeric columns to floats where possible
        df_input = pd.DataFrame(input_data)
        # Attempt to convert columns to numeric where possible
        for c in df_input.columns:
            df_input[c] = pd.to_numeric(df_input[c], errors="ignore")
    else:
        st.info("No feature list from model. You can upload a single-row CSV with the correct columns, or upload a CSV in batch mode.")
        uploaded_single = st.file_uploader("Upload a single-row CSV (optional)", type=["csv"])
        if uploaded_single:
            df_input = pd.read_csv(uploaded_single)
            if len(df_input) != 1:
                st.warning("Uploaded CSV has more than 1 row — only the first row will be used.")
                df_input = df_input.iloc[[0]]
        else:
            st.stop()

    if st.button("Predict (single)"):
        try:
            # Model may be a dict artifact containing "pipeline"
            predictor = model
            if isinstance(model, dict) and "pipeline" in model:
                predictor = model["pipeline"]
            proba = None
            if hasattr(predictor, "predict_proba"):
                proba = predictor.predict_proba(df_input)
                # assume binary classification, class 1 proba
                if proba.shape[1] >= 2:
                    score = proba[0, 1]
                else:
                    score = proba[0, 0]
                st.metric("Predicted probability (class 1)", f"{score:.3f}")
            pred = predictor.predict(df_input)
            st.write("Predicted label:", pred[0])
            # show input and prediction
            out = df_input.copy()
            out["prediction"] = pred
            if proba is not None:
                out["prob_class_1"] = float(score)
            st.dataframe(out.T)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:  # Batch mode - CSV
    st.subheader("Upload CSV for batch predictions")
    st.markdown("CSV should have the same columns used during training. If the model includes preprocessing inside a pipeline, columns order is not strict but column names should match.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        if st.button("Run batch prediction"):
            try:
                predictor = model
                if isinstance(model, dict) and "pipeline" in model:
                    predictor = model["pipeline"]
                preds = predictor.predict(df)
                out_df = df.copy()
                out_df["prediction"] = preds
                # add probabilities if available
                if hasattr(predictor, "predict_proba"):
                    probs = predictor.predict_proba(df)
                    if probs.shape[1] >= 2:
                        out_df["prob_class_1"] = probs[:, 1]
                    else:
                        out_df["prob_class_0"] = probs[:, 0]
                st.success("Predictions complete")
                st.dataframe(out_df.head(50))

                # allow download
                towrite = io.BytesIO()
                out_df.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("Download predictions as CSV", data=towrite, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# Footer / tips
st.markdown("---")
st.caption("If your model requires special preprocessing or custom feature encoders, ensure you saved the full pipeline (preprocessing + estimator) into the model file (joblib/pickle). If you need, I can help adapt this app to match your pipeline.")
