import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import os
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st
import joblib
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification

# Load Models
log_model = joblib.load('clickbait_log_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
lstm_model = load_model("clickbait_lstm_model.keras")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForSequenceClassification.from_pretrained("clickbait_bert_model").to(device)
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocess for all models
def preprocess_text(text):
    import re, string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()

# Streamlit Interface
st.title('Clickbait Headline Detector (Multi-Model)')
model_choice = st.selectbox("Choose the model to use:", ["Logistic Regression", "LSTM", "BERT"])
user_input = st.text_area("Enter news headline:")

if st.button("Check Headline"):
    processed_text = preprocess_text(user_input)
    if model_choice == "Logistic Regression":
        text_vectorized = vectorizer.transform([processed_text])
        prediction = log_model.predict(text_vectorized)[0]
    elif model_choice == "LSTM":
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts([processed_text])
        seq = tokenizer.texts_to_sequences([processed_text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
        prediction = (lstm_model.predict(padded)[0][0] > 0.5).astype("int32")
    elif model_choice == "BERT":
        inputs = tokenizer_bert(user_input, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    st.write(f"Prediction using {model_choice}: {'Clickbait' if prediction == 1 else 'Not Clickbait'}")