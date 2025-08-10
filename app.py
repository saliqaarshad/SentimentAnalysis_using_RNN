import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your trained model
model = tf.keras.models.load_model("imdb_sentiment_model.keras")

# Same parameters as training
vocab_size = 5000
max_length = 500
punc = r'''  !()-[]{};:'"\,<>./?@#$%^&*_~  '''

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    for ch in text:
        if ch in punc:
            text = text.replace(ch, " ")
    # One-hot encode and pad
    encoded = one_hot(text, vocab_size)
    padded = pad_sequences([encoded], maxlen=max_length, padding='post')
    return padded

st.title("IMDB Sentiment Analysis")

user_input = st.text_area("Enter a movie review text")

if st.button("Predict Sentiment"):
    if user_input.strip():
        processed = preprocess_text(user_input)
        prediction = model.predict(processed)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence: {prediction:.2f}")
    else:
        st.write("Please enter a review to analyze.")
