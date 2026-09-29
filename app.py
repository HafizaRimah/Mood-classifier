import streamlit as st
import sys
import os

# ---------------------------------------------------------------------------
# Setup: make sure Python can find the src/ folder so we can import predict.py
# ---------------------------------------------------------------------------
# When you run "streamlit run app.py" from the project root, the current
# working directory is the project root.  We add the "src" folder to Python's
# search path so that "from predict import ..." works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_emotion, labels  # reuse YOUR existing code


# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mood Classifier",
    page_icon="🎭",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Custom CSS for a clean, modern look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    /* Subtitle / description */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Predicted-emotion card */
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .emotion-card h2 {
        margin: 0;
        font-size: 2rem;
    }
    .emotion-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    /* Pipeline diagram */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin: 1.2rem 0;
    }
    .pipeline-step {
        background: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.45rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
    }
    .pipeline-arrow {
        font-size: 1.1rem;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🎭 Mood Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    'Type a sentence and discover the emotion behind it.<br>'
    'Powered by TF-IDF and Logistic Regression.'
    '</p>',
    unsafe_allow_html=True,
)

# Show the ML pipeline visually
st.markdown(
    '<div class="pipeline">'
    '<span class="pipeline-step">✏️ Your Text</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">📊 TF-IDF</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">🤖 Logistic Regression</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">🎯 Predicted Emotion</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Emoji map — one emoji per emotion for visual flair
# ---------------------------------------------------------------------------
EMOJI = {
    "sadness":  "😢",
    "joy":      "😄",
    "love":     "❤️",
    "anger":    "😠",
    "fear":     "😨",
    "surprise": "😲",
}

# Color for each emotion's progress bar (Streamlit accepts CSS-style colors)
COLOR = {
    "sadness":  "#6366f1",   # indigo
    "joy":      "#eab308",   # yellow
    "love":     "#ec4899",   # pink
    "anger":    "#ef4444",   # red
    "fear":     "#8b5cf6",   # purple
    "surprise": "#f97316",   # orange
}


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
user_text = st.text_area(
    "Enter a sentence to analyze:",
    height=120,
    placeholder="e.g.  I finally achieved my goal and I am so happy!",
)

analyze_button = st.button("🔍 Analyze Mood", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Prediction & results
# ---------------------------------------------------------------------------
if analyze_button:

    # --- Guard: empty input ---
    if not user_text.strip():
        st.warning("⚠️ Please enter a sentence before analyzing.")
    else:
        try:
            # Call YOUR existing prediction function
            predicted_emotion, probabilities = predict_emotion(user_text)

            # ---- Predicted emotion card ----
            emoji = EMOJI.get(predicted_emotion, "🎭")
            st.markdown(
                f'<div class="emotion-card">'
                f'<h2>{emoji} {predicted_emotion.capitalize()}</h2>'
                f'<p>Predicted emotion</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ---- Probability breakdown ----
            st.subheader("📊 Emotion Probabilities")

            # Sort emotions by probability (highest first) for readability
            sorted_indices = sorted(
                range(len(labels)),
                key=lambda i: probabilities[i],
                reverse=True,
            )

            for idx in sorted_indices:
                emotion_name = labels[idx]
                prob = probabilities[idx]
                emoji_icon = EMOJI.get(emotion_name, "")
                is_predicted = emotion_name == predicted_emotion

                # Highlight the predicted emotion
                label_text = f"{emoji_icon} **{emotion_name.capitalize()}**" if is_predicted else f"{emoji_icon} {emotion_name.capitalize()}"

                weight = "700" if is_predicted else "400"

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.progress(float(prob), text=label_text)
                with col2:
                    st.markdown(
                        f"<p style='text-align:right; font-size:1.1rem; "
                        f"margin-top:0.3rem; font-weight:{weight}'>"
                        f"{prob:.1%}</p>",
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"❌ Something went wrong during prediction: {e}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Mood Classifier · Logistic Regression + TF-IDF · Built with Streamlit")
