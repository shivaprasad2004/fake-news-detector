import os
import requests
import streamlit as st
from lime.lime_text import LimeTextExplainer
from dotenv import load_dotenv

# =========================
# 🔒 Load API Keys securely from .env
# =========================
load_dotenv()  # loads .env file automatically

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# =========================
# Hugging Face Query
# =========================
def query_hf_model(text):
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": ["real", "fake"]},
    }
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": f"HF API Error {response.status_code}: {response.text}"}

    try:
        return response.json()
    except Exception as e:
        return {"error": f"JSON decode error: {str(e)} | Raw: {response.text}"}

# =========================
# Google Fact Check Query
# =========================
def query_google_factcheck(text):
    params = {"query": text, "key": GOOGLE_FACTCHECK_API_KEY}
    response = requests.get(GOOGLE_FACTCHECK_URL, params=params)
    return response.json()

# =========================
# Prediction Function
# =========================
def predict_text_authenticity(text):
    result = query_hf_model(text)

    if "error" in result:
        return result["error"], 0.0

    if isinstance(result, dict) and "labels" in result:
        label = result["labels"][0]
        score = result["scores"][0]
        return label, score

    return "Unknown", 0.0

# =========================
# Render Prediction Style
# =========================
def render_prediction(prediction, score):
    styles = {
        'fake': {"background": "#ff4c4c", "text": "#fff"},
        'real': {"background": "#4caf50", "text": "#fff"},
        'unknown': {"background": "#6c757d", "text": "#fff"}
    }
    style = styles.get(prediction.lower(), styles["unknown"])

    st.markdown(
        f"""
        <div style="
            background: {style['background']};
            color: {style['text']};
            padding: 25px;
            border-radius: 20px;
            border: 2px solid #444;
            box-shadow: 0 0 15px rgba(255,255,255,0.2);
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 25px;
        ">
            🔮 Prediction: <b>{prediction.upper()}</b><br>
            🎯 Confidence: {score:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Main Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

    # Custom Dark Theme
    st.markdown("<h1 style='text-align:center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>AI + Fact-Check Verification Dashboard</h3>", unsafe_allow_html=True)

    user_input = st.text_area("✍️ Enter a sentence to predict its truthfulness:")

    if st.button("🚀 Verify News"):
        if not HF_API_TOKEN or not GOOGLE_FACTCHECK_API_KEY:
            st.error("⚠️ API keys not found. Please create a .env file.")
            return

        # 1. Predict using Hugging Face model
        prediction, score = predict_text_authenticity(user_input)
        render_prediction(prediction, score)

        # 2. Explain with LIME
        explainer = LimeTextExplainer(class_names=["real", "fake"])
        def classifier_fn(texts):
            results = [predict_text_authenticity(t)[1] or 0.5 for t in texts]
            return [[1-r, r] for r in results]
        explanation = explainer.explain_instance(user_input, classifier_fn, num_features=10)
        st.markdown("### 🧠 Explanation of Prediction (LIME)")
        st.pyplot(explanation.as_pyplot_figure())

        # 3. Google Fact Check
        st.markdown("### 🔎 Fact-Check Results from Google")
        fact_results = query_google_factcheck(user_input)
        if "claims" in fact_results:
            for claim in fact_results["claims"]:
                text = claim.get("text", "N/A")
                rating = claim["claimReview"][0].get("textualRating", "N/A")
                source = claim["claimReview"][0].get("publisher", {}).get("name", "Unknown")
                url = claim["claimReview"][0].get("url", "#")
                st.markdown(
                    f"<div style='background:#1E1E1E;padding:12px;border-radius:8px;margin:8px 0;border-left:4px solid #BB86FC;'>"
                    f"<b>{text}</b><br>"
                    f"➡️ {rating} (Source: <a href='{url}' target='_blank'>{source}</a>)"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No fact-check results found.")

if __name__ == "__main__":
    main()
