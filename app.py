import streamlit as st
import pickle
import re
import scipy.sparse as sp
import numpy as np

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# ----------------------------
# FEATURE EXPLANATION 
# ----------------------------
def explain_prediction(features, tfidf_vec, title, body):
    """Explain why the model made its decision using feature contributions."""
    feature_names = list(tfidf.get_feature_names_out()) + ["text_length", "exclamation_count", "uppercase_ratio"]
    
    coef = model.coef_[0]
    
    tfidf_array = tfidf_vec.toarray()[0]
    text = title + " " + body
    tabular_values = [len(text), body.count("!"), 
                      sum(1 for c in text if c.isupper()) / len(text) if len(text)>0 else 0]
    
    all_features = tfidf_array.tolist() + tabular_values
    
    contributions = []
    for i, (name, value, coeff) in enumerate(zip(feature_names, all_features, coef)):
        if value != 0: 
            contrib = value * coeff
            contributions.append((name, contrib, coeff))
    
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return contributions, feature_names

# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def build_features(title, body):
    text = title + " " + body
    clean = clean_text(text)

    text_length = len(text)
    exclamation_count = text.count("!")

    uppercase_ratio = (
        sum(1 for c in text if c.isupper()) / len(text)
        if len(text) > 0 else 0
    )

    tfidf_vec = tfidf.transform([clean])

    features = sp.hstack((
        tfidf_vec,
        [[text_length, exclamation_count, uppercase_ratio]]
    ))

    return features, tfidf_vec

# ----------------------------
# UI DESIGN
# ----------------------------
st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>🧠 Fake News Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>Analyze news articles using AI-powered detection</p>",
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# INPUT SECTION
# ----------------------------
title = st.text_input("📰 Article Title")

body = st.text_area("📄 Article Content", height=200)

# ----------------------------
# BUTTON
# ----------------------------
if st.button("🔍 Analyze Article"):
    if title and body:

        features, tfidf_vec = build_features(title, body)
        prob_fake = model.predict_proba(features)[0][0]
        st.divider()

        # ----------------------------
        # RESULT SECTION
        # ----------------------------
        st.subheader("📊 Analysis Result")

        st.metric(
            label="Fake News Probability",
            value=f"{prob_fake*100:.2f}%"
        )

        st.progress(float(prob_fake))

        if prob_fake > 0.75:
            st.error("🚨 High Risk: Likely Fake News")
        elif prob_fake > 0.5:
            st.warning("⚠️ Medium Risk: Suspicious Content")
        else:
            st.success("✅ Low Risk: Likely Real News")

        st.divider()

        # ----------------------------
        # FEATURE INSIGHT 
        # ----------------------------
        st.subheader("🔎 What the AI looked at")

        col1, col2, col3 = st.columns(3)

        text_len = len(title + body)
        exclam = body.count("!")
        caps_ratio = sum(1 for c in (title + body) if c.isupper()) / len(title + body) if len(title+body)>0 else 0

        with col1:
            st.metric("Text Length", text_len)

        with col2:
            st.metric("Exclamation Marks", exclam)

        with col3:
            st.metric("Caps Ratio", f"{caps_ratio:.2f}")

        # ----------------------------
        # FEATURE EXPLANATION
        # ----------------------------
        st.markdown("---")

        contributions, feat_names = explain_prediction(features, tfidf_vec, title, body)

        positive_contrib = []
        negative_contrib = []

        for name, contrib, coeff in contributions:
            if contrib > 0:
                negative_contrib.append((name, contrib))
            elif contrib < 0:
                positive_contrib.append((name, contrib))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🚨 Signs of Fake News")
            if positive_contrib:
                max_contrib = max(abs(v) for _, v in positive_contrib[:5]) if positive_contrib else 1
                for name, val in positive_contrib[:5]:
                    strength = (abs(val) / max_contrib * 100) if max_contrib > 0 else 0
                    st.write(f"**\"{name}\"**")
                    st.progress(strength / 100)
                    st.markdown("")
            else:
                st.write("No strong indicators found")

        with col2:
            st.markdown("### ✅ Signs of Real News")
            if negative_contrib:
                max_contrib = max(abs(v) for _, v in negative_contrib[:5]) if negative_contrib else 1
                for name, val in negative_contrib[:5]:
                    strength = (abs(val) / max_contrib * 100) if max_contrib > 0 else 0
                    st.write(f"**\"{name}\"**")
                    st.progress(strength / 100)
                    st.markdown("")
            else:
                st.write("No strong indicators found")

        st.markdown("---")
        with st.expander("ℹ️ How to interpret this analysis"):
            st.write("""
            **How this works:**
            - The AI compares this article to thousands of real and fake news examples
            - Words and phrases are scored based on whether they appear more often in fake or real news
            - The final score is the sum of all these factors
            
            **What to look for:**
            - Fake news often uses sensational words, emotional language, and clickbait phrases
            - Real news typically uses factual, neutral language and proper sourcing
            - No single word determines the result - it's the combination of all factors
            """)

    else:
        st.warning("⚠️ Please enter both title and article content")