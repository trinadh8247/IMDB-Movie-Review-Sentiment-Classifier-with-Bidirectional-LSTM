import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model


max_features = 10000  # must match training vocabulary size
maxlen = 300          # must match training sequence length

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

model = load_model('imdb_rnn_model.h5')

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_text(text):
    words = text.lower().split()
    indices = []
    for word in words:
        idx = word_index.get(word)
        # Map unseen or out-of-vocabulary tokens to the reserved OOV index (2)
        if idx is None or idx >= max_features:
            idx = 2
        indices.append(idx + 3)  # offset by 3 to align with IMDB encoding
    return tf.keras.preprocessing.sequence.pad_sequences([indices], maxlen=maxlen)

## prediction function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text, verbose=0)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## Streamlit app with professional UI
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .header-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subheader {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-title">🎬 IMDB Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Powered by Bidirectional LSTM Neural Network</div>', unsafe_allow_html=True)

st.divider()

# Main content
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📝 Enter Your Movie Review")
    user_input = st.text_area(
        "Paste or type a movie review below:",
        placeholder="e.g., 'This movie was absolutely amazing! Best film I've seen all year.'",
        height=150,
        label_visibility="collapsed",
        key="review_input"
    )
    
    # Prediction button
    if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
        if user_input.strip():
            st.session_state.user_input = user_input
            st.session_state.show_results = True
        else:
            st.warning("⚠️ Please enter a review to analyze!")
    
with col2:
    st.subheader("⚡ Quick Test")
    sample_reviews = {
        "Positive": "This movie was absolutely fantastic! Best film I've seen in years.",
        "Negative": "Terrible waste of time. Horrible acting and boring plot.",
        "Mixed": "Loved the performances, but the plot was a mess."
    }
    
    for label, review in sample_reviews.items():
        if st.button(f"Test: {label}", use_container_width=True):
            st.session_state.user_input = review
            st.session_state.show_results = True

# Results section
if st.session_state.show_results and st.session_state.user_input.strip():
    user_input = st.session_state.user_input
    with st.spinner("⏳ Analyzing sentiment..."):
        sentiment, confidence = predict_sentiment(user_input)
    
    # Convert confidence to Python float for Streamlit compatibility
    confidence = float(confidence)
    
    st.divider()
    st.subheader("📊 Prediction Results")
    
    # Display sentiment with colors
    if sentiment == "Positive":
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown(
                f"""
                <div class="sentiment-positive">
                <h3>✅ POSITIVE SENTIMENT</h3>
                <p>This review expresses favorable opinions about the movie.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.metric("Confidence Score", f"{confidence*100:.1f}%", delta=f"{(confidence-0.5)*100:.1f}%")
            st.progress(confidence, text=f"{confidence*100:.1f}%")
    else:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown(
                f"""
                <div class="sentiment-negative">
                <h3>❌ NEGATIVE SENTIMENT</h3>
                <p>This review expresses unfavorable opinions about the movie.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.metric("Confidence Score", f"{(1-confidence)*100:.1f}%", delta=f"{(0.5-confidence)*100:.1f}%")
            st.progress(1 - confidence, text=f"{(1-confidence)*100:.1f}%")
    
    # Review analysis
    st.subheader("📌 Review Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", len(user_input.split()))
    with col2:
        st.metric("Character Count", len(user_input))
    with col3:
        st.metric("Sentiment", sentiment)

st.divider()

# Footer
st.markdown("""
    <div style="text-align: center; color: #999; margin-top: 3rem; font-size: 0.85rem;">
    <p>🧠 Model: Bidirectional LSTM with Dropout | 📊 Dataset: IMDB Reviews (75K samples) | 🎯 Accuracy: ~91%</p>
    <p><small>Built with Streamlit | Trained with TensorFlow/Keras</small></p>
    </div>
""", unsafe_allow_html=True)
