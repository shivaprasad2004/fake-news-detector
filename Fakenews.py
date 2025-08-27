import os
import requests
import streamlit as st
from lime.lime_text import LimeTextExplainer
from dotenv import load_dotenv

# =========================
# 🔒 Load API Keys
# =========================
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

HF_MODEL_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-1"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# =========================
# HuggingFace Model
# =========================
def query_hf_model(text):
    payload = {"inputs": text, "parameters": {"candidate_labels": ["real", "fake"]}}
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return {"error": f"HF API Error {response.status_code}: {response.text}"}
    return response.json()

# =========================
# Google Fact Check
# =========================
def query_google_factcheck(text):
    params = {"query": text, "key": GOOGLE_FACTCHECK_API_KEY}
    response = requests.get(GOOGLE_FACTCHECK_URL, params=params)
    return response.json()

# =========================
# News API + Fallback (Tavily / Google News)
# =========================
def query_newsapi(text):
    params = {
        "q": text,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 3
    }
    try:
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        if "articles" in data and len(data["articles"]) > 0:
            return data
        else:
            # fallback → Google News
            alt_url = f"https://news.google.com/search?q={text.replace(' ', '+')}"
            return {"results": [{"title": "Google News Search", "url": alt_url, "source": "Google News"}]}
    except Exception as e:
        return {"error": str(e)}

# =========================
# Prediction
# =========================
def predict_text_authenticity(text):
    result = query_hf_model(text)
    if "error" in result:
        return "fake", 0.0

    if isinstance(result, dict) and "labels" in result:
        label = result["labels"][0].lower()
        score = result["scores"][0]
        if label not in ["real", "fake"]:
            label = "real" if score >= 0.5 else "fake"
        return label, score

    return "fake", 0.0

# =========================
# Streamlit UI
# =========================
def render_prediction(prediction, score, sources):
    styles = {
        'fake': {"background": "#ff4c4c", "text": "#fff"},
        'real': {"background": "#4caf50", "text": "#fff"},
        'uncertain': {"background": "#6c757d", "text": "#fff"}
    }
    style = styles.get(prediction.lower(), styles["uncertain"])

    st.markdown(
        f"""
        <div style="
            background: {style['background']};
            color: {style['text']};
            padding: 25px;
            border-radius: 20px;
            border: 2px solid #444;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        ">
            🔮 Prediction: <b>{prediction.upper()}</b><br>
            🎯 Confidence: {score:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔎 Verified Sources")
    if not sources:
        st.info("No sources found.")
    else:
        for idx, (title, url, src) in enumerate(sources, 1):
            st.markdown(f"**{idx}. {src} →** [{title}]({url})")

# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
    st.markdown("<h1 style='text-align:center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>AI + Multi-Source Fact Verification</h3>", unsafe_allow_html=True)

    user_input = st.text_area("✍️ Enter a news headline or claim:")

    if st.button("🚀 Verify News"):
        if not HF_API_TOKEN or not GOOGLE_FACTCHECK_API_KEY or not NEWS_API_KEY:
            st.error("⚠️ Missing API Keys in .env file.")
            return

        # 1. Prediction
        prediction, score = predict_text_authenticity(user_input)

        # 2. Collect Sources
        sources = []

        # Google Fact Check
        fact_results = query_google_factcheck(user_input)
        if "claims" in fact_results:
            for claim in fact_results["claims"][:3]:
                text = claim.get("text", "N/A")
                review = claim["claimReview"][0]
                url = review.get("url", "#")
                source = review.get("publisher", {}).get("name", "Unknown")
                sources.append((text, url, source))
                print(f"[FACTCHECK] {source} → {url}")  # console log

        # News API (with fallback)
        news_results = query_newsapi(user_input)
        if "articles" in news_results:
            for article in news_results["articles"][:3]:
                title = article.get("title", "N/A")
                url = article.get("url", "#")
                source = article["source"].get("name", "Unknown")
                sources.append((title, url, source))
                print(f"[NEWSAPI] {source} → {url}")
        elif "results" in news_results:
            for res in news_results["results"][:3]:
                title = res.get("title", "N/A")
                url = res.get("url", "#")
                source = res.get("source", "Unknown")
                sources.append((title, url, source))
                print(f"[FALLBACK] {source} → {url}")

        # 3. Always guarantee at least 1 source
        if not sources:
            fallback_url = f"https://news.google.com/search?q={user_input.replace(' ', '+')}"
            sources.append(("Search on Google News", fallback_url, "Google News"))

        # 4. Show result
        render_prediction(prediction, score, sources)

if __name__ == "__main__":
    main()
