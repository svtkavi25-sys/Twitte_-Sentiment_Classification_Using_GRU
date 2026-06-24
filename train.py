
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("dataset/tweets.csv")

# Texts
texts = data["tweet"].astype(str)

# Clean labels
labels = data["sentiment"].astype(str).str.strip().str.lower()

# Label mapping
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# Convert labels
labels = labels.map(label_map)

# Remove invalid rows
valid_indices = labels.notna()

texts = texts[valid_indices]
labels = labels[valid_indices]

# Convert labels to integer
labels = labels.astype(int)

# Tokenization settings
max_words = 5000
max_len = 20

# Tokenizer
tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Padding
X = pad_sequences(sequences, maxlen=max_len)

# Convert labels to categorical
y = to_categorical(labels, num_classes=3)

# Shuffle dataset
indices = np.arange(len(X))

np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Train-test split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Build GRU model
model = Sequential([
    
    Embedding(max_words, 128, input_length=max_len),

    GRU(
        128,
        dropout=0.2,
        recurrent_dropout=0.2
    ),

    Dense(64, activation='relu'),

    Dense(3, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model/sentiment_gru.h5")

# Save tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model trained and saved successfully!")

