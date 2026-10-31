
import streamlit as st
import joblib
import os
import json
import re

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VerifiAI",
    page_icon="🔎",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #14213d, #176b87);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #e0f2fe;
    font-size: 16px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-top: 15px;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

MODEL_PATH = "models/fake_news_model.joblib"
METRICS_PATH = "models/evaluation_metrics.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🔎 VerifiAI")

    st.write("AI-powered News Analysis")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        ["News Analyzer", "About Project"]
    )

    st.divider()

    st.caption("Powered by Machine Learning")
    st.caption("Python | Scikit-learn | Streamlit")


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🔎 VerifiAI</h1>
    <p>
        Analyze news articles using Machine Learning.
        Explore patterns associated with fake and real news.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- CHECK MODEL ----------------

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = load_model()


# ---------------- NEWS ANALYZER ----------------

if page == "News Analyzer":

    st.subheader("📰 News Article Analyzer")

    st.write(
        "Paste a news article below and let the trained "
        "machine learning model analyze its text."
    )

    article = st.text_area(
        "Enter News Article",
        placeholder="Paste your news headline and article here...",
        height=220
    )

    col1, col2 = st.columns([1, 5])

    with col1:
        analyze = st.button(
            "Analyze News 🔍",
            type="primary",
            use_container_width=True
        )

    if analyze:

        if len(article.strip()) < 30:
            st.warning(
                "Please enter at least 30 characters "
                "of news text."
            )

        else:
            try:
                prediction = model.predict([article])[0]

                probabilities = model.predict_proba([article])[0]

                classes = list(model.classes_)

                fake_index = classes.index(0)
                real_index = classes.index(1)

                fake_probability = probabilities[fake_index]
                real_probability = probabilities[real_index]

                confidence = max(probabilities) * 100

                st.divider()

                st.subheader("📊 Analysis Report")

                result_col, confidence_col = st.columns(2)

                with result_col:
                    if prediction == 0:
                        st.error("🚨 Prediction: FAKE NEWS")
                    else:
                        st.success("✅ Prediction: REAL NEWS")

                with confidence_col:
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                # Probability breakdown

                st.subheader("Prediction Probability")

                fake_col, real_col = st.columns(2)

                with fake_col:
                    st.write("🚨 Fake News")
                    st.progress(float(fake_probability))
                    st.write(
                        f"{fake_probability * 100:.2f}%"
                    )

                with real_col:
                    st.write("✅ Real News")
                    st.progress(float(real_probability))
                    st.write(
                        f"{real_probability * 100:.2f}%"
                    )

                # Article statistics

                st.divider()

                st.subheader("📝 Article Statistics")

                words = re.findall(
                    r"\b[\w'-]+\b",
                    article
                )

                characters = len(article)
                word_count = len(words)
                sentence_count = len(
                    [s for s in re.split(r'[.!?]+', article) if s.strip()]
                )

                c1, c2, c3 = st.columns(3)

                c1.metric("Words", word_count)
                c2.metric("Characters", characters)
                c3.metric("Sentences", sentence_count)

                st.info(
                    "This result is a machine learning prediction "
                    "based on text patterns. It does not verify "
                    "whether the article is factually true."
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.subheader("About VerifiAI")

    st.write("""
    TruthLens AI is a machine learning-based news
    classification application.

    It analyzes article text and predicts whether the
    text resembles examples labelled as fake or real
    in the training dataset.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorizer
    - Logistic Regression
    """)

    st.subheader("How It Works")

    st.markdown("""
    1. User enters a news article.
    2. TF-IDF converts text into numerical features.
    3. Logistic Regression analyzes those features.
    4. The application displays the predicted class
       and model probabilities.
    """)

    st.subheader("Model Performance")

    if os.path.exists(METRICS_PATH):

        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Training Samples",
            metrics["training_samples"]
        )

        c3.metric(
            "Testing Samples",
            metrics["testing_samples"]
        )

    else:
        st.warning("Model evaluation metrics not found.")


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<p class="small-note">
VerifiAI | Fake News & Misinformation Detector
<br>
Always verify important information using reliable sources.
</p>
""", unsafe_allow_html=True)
