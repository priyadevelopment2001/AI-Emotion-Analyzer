import streamlit as st
from transformers import pipeline
import base64

st.set_page_config(
    page_title="AI Emotion Analyzer",
    page_icon="AI",
    layout="centered"
)

# Photo load
with open("priya_photo.jpeg", "rb") as f:
    photo = base64.b64encode(f.read()).decode()

# Styling
st.markdown("""
<style>

.main {
    padding-top: 0rem;
}

.header {
    text-align: center;
    padding: 10px;
}

.header h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.header p {
    font-size: 18px;
    color: #666;
}

.name {
    position: fixed;
    top: 15px;
    right: 25px;
    font-size: 17px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# PHOTO — TOP
st.markdown(
    f"""
    <div style="text-align:center; margin-top:0px; margin-bottom:10px;">
        <img src="data:image/jpeg;base64,{photo}"
             width="180"
             style="border-radius:15px;">
    </div>
    """,
    unsafe_allow_html=True
)


# Name
st.markdown(
    '<div class="name">Priya Dalal</div>',
    unsafe_allow_html=True
)


# Header
st.markdown("""
<div class="header">
    <h1>AI Emotion Analyzer</h1>
    <p>Understand emotions from your text using Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)


# AI Model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion_analyzer = load_model()


# Input
text = st.text_area(
    "Enter your text",
    placeholder="Example: I am very happy today because everything went well.",
    height=150
)


# Analyze
if st.button("Analyze Emotion", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text first.")

    else:
        result = emotion_analyzer(text)[0]

        emotion = result["label"]
        confidence = result["score"] * 100

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader("Analysis Result")

        st.success(f"Emotion: {emotion.upper()}")

        st.progress(result["score"])

        st.write(f"Confidence: **{confidence:.2f}%**")

        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")

st.caption(
    "AI Emotion Analyzer | Developed by Priya Dalal"
)
