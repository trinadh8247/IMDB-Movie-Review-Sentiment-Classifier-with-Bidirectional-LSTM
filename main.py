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

##streamlit app
import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")
user_input = st.text_area("Enter your movie review here:")
if st.button("Predict Sentiment"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence: **{confidence:.2f}**")
    else:
        st.write("Please enter a movie review to analyze.")
