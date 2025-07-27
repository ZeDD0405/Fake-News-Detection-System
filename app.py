import streamlit as st
import joblib

# ---- Load model and vectorizer ----
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---- Custom CSS for animations and styling ----
st.markdown("""
    <style>
    body {
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stTextArea > div > textarea {
        transition: all 0.3s ease;
        border-radius: 10px !important;
        border: 2px solid #ccc !important;
    }

    .stTextArea > div > textarea:hover {
        border: 2px solid #1f77b4 !important;
        box-shadow: 0 0 10px rgba(31, 119, 180, 0.3);
    }

    .stButton>button {
        background-color: #1f77b4;
        color: white;
        padding: 0.6em 1.2em;
        font-size: 1.1em;
        border-radius: 10px;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        background-color: #145a86;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .stMarkdown {
        font-size: 1.1em;
    }

    footer {
        text-align: center;
        font-size: 0.9em;
        color: #999;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---- App title ----
st.title("📰 Fake News Detection App")
st.markdown("Check if a news article is **Real** or **Fake** using a trained ML model.")

# ---- Text input ----
news_text = st.text_area("📝 Enter News Text Here:", height=250)

# ---- Predict Button ----
if st.button("🚀 Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text to check.")
    else:
        vector = vectorizer.transform([news_text])
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][prediction]

        if prediction == 1:
            st.success(f"✅ This news is **REAL**. (Confidence: {prob*100:.2f}%)")
        else:
            st.error(f"❌ This news is **FAKE**. (Confidence: {prob*100:.2f}%)")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<footer>🚀 Developed and Deployed by <strong>Sagar Kallimani</strong> using Streamlit</footer>",
    unsafe_allow_html=True
)
