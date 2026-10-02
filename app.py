import streamlit as st
from transformers import pipeline
import base64

# Page settings
st.set_page_config(
    page_title="AI Emotion Analyzer",
    page_icon="AI",
    layout="centered"
)

# Profile Photo
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


# Profile Photo
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


# Developer name
st.markdown(
    '<div class="name">Priya Dalal</div>',
    unsafe_allow_html=True
)


# Header
st.markdown("""
<div class="header">
    <h1>AI Emotion Analyzer</h1>
    <p>AI-powered emotion detection from text</p>
</div>
""", unsafe_allow_html=True)


# Load AI model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-emotion-classification",
        function_to_apply="sigmoid",
        top_k=None
    )


emotion_analyzer = load_model()


# Text Input
text = st.text_area(
    "Enter your text",
    placeholder="Example: Mujhe aaj bahut khushi ho rahi hai!",
    height=150
)


# Analyze Emotion
if st.button("Analyze Emotion", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text first.")

    else:

        results = emotion_analyzer(text)

        # Handle model output
        if results and isinstance(results[0], list):
            results = results[0]

        # Sort by confidence
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        # Highest emotion
        best_result = results[0]

        emotion = best_result["label"]
        confidence = best_result["score"] * 100

        # Main Result
        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader("Analysis Result")

        st.success(
            f"Primary Emotion: {emotion.upper()}"
        )

        st.progress(
            float(best_result["score"])
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # Emotion Details
        st.subheader("Emotion Details")

        for item in results[:5]:

            label = item["label"]
            score = item["score"] * 100

            st.write(
                f"**{label.upper()}** — {score:.2f}%"
            )

            st.progress(
                float(item["score"])
            )


# Footer
st.markdown("---")

st.caption(
    "AI Emotion Analyzer | Developed by Priya Dalal"
)
