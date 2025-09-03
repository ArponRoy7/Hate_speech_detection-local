# streamlit_app.py — minimal UI (only prediction, no notes/overrides)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # quieter TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st

st.set_page_config(page_title="Hate Speech Detection", page_icon="🚨")
st.title("🚨 Hate Speech Detection (BERT+LSTM+CNN)")

# Safe import so errors show on the page (avoid blank screen)
try:
    from hsd_model import ensure_model, predict_label, predict_proba
except Exception as e:
    st.error(f"Import error while loading modules:\n\n{e}")
    st.stop()

@st.cache_resource
def _bootstrap_safe():
    try:
        return ensure_model()
    except Exception as e:
        return e  # return exception to render visibly

bootstrap = _bootstrap_safe()
if isinstance(bootstrap, Exception):
    st.error(f"Startup error in ensure_model():\n\n{bootstrap}")
    st.stop()

st.markdown("Paste text below and click **Predict**.")

# ── Main input & actions ──
text = st.text_area("Enter text:", placeholder="Type something...", height=140)

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Predict", type="primary")
with col2:
    probs_cb = st.checkbox("Show class probabilities", value=True)

# Color mapping for UI
COLOR_HEX = {
    "offensive": "#e74c3c",     # red
    "hate speech": "#f1c40f",   # yellow
    "neutral": "#2ecc71",       # green
}

def show_prediction_badge(label: str):
    if label == "offensive":
        st.error(f"Prediction: **{label}**")
    elif label == "hate speech":
        st.warning(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

def bar_row(label: str, pct: float):
    pct = max(0.0, min(100.0, pct))
    color = COLOR_HEX.get(label, "#888")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
          <div style="width:130px;font-weight:600;color:{color};text-transform:capitalize;">
            {label}
          </div>
          <div style="flex:1;height:12px;background:#e9ecef;border-radius:6px;overflow:hidden;">
            <div style="width:{pct:.2f}%;height:100%;background:{color};"></div>
          </div>
          <div style="width:70px;text-align:right;font-variant-numeric:tabular-nums;">
            {pct:.2f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_btn:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            label = predict_label(text.strip())
            show_prediction_badge(label)

            if probs_cb:
                probs = predict_proba(text.strip())  # dict label -> prob (0..1)
                st.subheader("Probabilities (sorted):")
                for lbl, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
                    bar_row(lbl, p * 100.0)
        except Exception as e:
            st.error(f"Inference error:\n\n{e}")
