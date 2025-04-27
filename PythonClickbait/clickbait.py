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

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Dataset
df = pd.read_csv("clickbait_data.csv")
df.columns = ["Headline", "Label"]
df['Label'] = df['Label'].map({"Clickbait": 1, "Not Clickbait": 0})
df.dropna(subset=['Label', 'Headline'], inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing Function
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()

df['cleaned_headline'] = df['Headline'].apply(preprocess_text)
df = df[df['cleaned_headline'].str.len() > 0]  # Ensure non-empty rows

# Feature Extraction
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['cleaned_headline'])
y = df['Label']
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# BERT Model
from torch.optim import AdamW
from sklearn.metrics import classification_report

tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Prepare inputs for training
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

encoded_inputs = tokenizer_bert(list(df['Headline']), padding=True, truncation=True, return_tensors='pt', max_length=64)
X_train_ids, X_test_ids, y_train_bert, y_test_bert = train_test_split(
    encoded_inputs['input_ids'], df['Label'].values, test_size=0.2, random_state=42)
X_train_mask, X_test_mask = train_test_split(encoded_inputs['attention_mask'], test_size=0.2, random_state=42)

dataset_train = TensorDataset(X_train_ids, X_train_mask, torch.tensor(y_train_bert))
dataset_test = TensorDataset(X_test_ids, X_test_mask, torch.tensor(y_test_bert))
train_loader = DataLoader(dataset_train, batch_size=8)
test_loader = DataLoader(dataset_test, batch_size=8)

# Training loop
bert_model.train()
optimizer = AdamW(bert_model.parameters(), lr=2e-5)

for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        outputs = bert_model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
bert_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        outputs = bert_model(input_ids=b_input_ids, attention_mask=b_input_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        y_true.extend(b_labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

print("\nClassification Report (BERT):")
print(classification_report(y_true, y_pred, target_names=["Non-Clickbait", "Clickbait"]))

bert_model.save_pretrained('clickbait_bert_model')

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train_tfidf, y_train)
joblib.dump(log_model, "clickbait_log_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# LSTM Model
max_words = 5000
max_len = 100

tokenizer_lstm = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer_lstm.fit_on_texts(df['cleaned_headline'])
X_seq = tokenizer_lstm.texts_to_sequences(df['cleaned_headline'])
X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_len)
X_train_pad, X_test_pad, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

lstm_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))
lstm_model.save("clickbait_lstm_model.keras")


# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = [1 if p > 0.5 else 0 for p in y_pred]
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    return accuracy, precision, recall, f1

log_acc, log_prec, log_rec, log_f1 = evaluate_model(log_model, X_test_tfidf, y_test)
lstm_acc, lstm_prec, lstm_rec, lstm_f1 = evaluate_model(lstm_model, X_test_pad, y_test)

print(f"Logistic Regression - Accuracy: {log_acc}, Precision: {log_prec}, Recall: {log_rec}, F1-Score: {log_f1}")
print(f"LSTM - Accuracy: {lstm_acc}, Precision: {lstm_prec}, Recall: {lstm_rec}, F1-Score: {lstm_f1}")


