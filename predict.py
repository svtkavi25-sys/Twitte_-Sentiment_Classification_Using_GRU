import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("model/sentiment_gru.h5")

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Labels
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

max_len = 20

while True:
    text = input("Enter Tweet: ")

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(seq, maxlen=max_len)

    prediction = model.predict(padded)

    label = np.argmax(prediction)

    print("Sentiment:", label_map[label])