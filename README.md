# Twitter Sentiment Classification using GRU

## Overview

This project is a Deep Learning NLP application that classifies tweet text into:

* Positive 😊
* Negative 😡
* Neutral 😐

The model is built using a GRU (Gated Recurrent Unit) neural network with TensorFlow and Keras.

The project uses a small dataset for fast training and beginner-friendly implementation.

---

# Features

* Text sentiment classification
* GRU-based deep learning model
* NLP preprocessing
* Tokenization and padding
* Multi-class classification
* Fast training on small dataset
* Real-time prediction using user input

---

# Technologies Used

* Python
* TensorFlow
* Keras
* Pandas
* NumPy

---

# Project Structure

```plaintext
twitter_gru_sentiment/
│
├── dataset/
│   └── tweets.csv
│
├── model/
│   ├── sentiment_gru.h5
│   └── tokenizer.pkl
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# Dataset Format

```csv
tweet,sentiment
I love this app,positive
Worst product ever,negative
The meeting starts at 5 PM,neutral
```

---

# How It Works

## Step 1 — Load Dataset

The dataset containing tweet text and sentiment labels is loaded using Pandas.

## Step 2 — Text Preprocessing

The text is cleaned and converted into lowercase format.

## Step 3 — Tokenization

Words are converted into numerical tokens using a tokenizer.

Example:

```text
I love this app
```

becomes:

```text
[1, 5, 8, 12]
```

## Step 4 — Padding

All sequences are padded to the same length for neural network processing.

## Step 5 — Embedding Layer

The embedding layer converts words into dense vector representations.

## Step 6 — GRU Layer

The GRU neural network learns sentiment patterns from the text sequences.

## Step 7 — Output Prediction

The model predicts one of the following sentiments:

* Positive
* Negative
* Neutral

---

# Model Architecture

```text
Input Text
     ↓
Tokenizer
     ↓
Padding
     ↓
Embedding Layer
     ↓
GRU Layer
     ↓
Dense Layer
     ↓
Softmax Output
     ↓
Sentiment Prediction
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/twitter_gru_sentiment.git
```

## Navigate to Project Folder

```bash
cd twitter_gru_sentiment
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python train.py
```

The trained model will be saved inside the `model/` folder.

---

# Run Prediction

```bash
python predict.py
```

Example:

```text
Enter Tweet: I really love this app
Sentiment: Positive
```

---

# Example Predictions

| Input Text                | Prediction |
| ------------------------- | ---------- |
| I love this app           | Positive   |
| Worst experience ever     | Negative   |
| The train arrives at 5 PM | Neutral    |

---

# Why GRU?

GRU (Gated Recurrent Unit) is used because it:

* Trains faster than LSTM
* Uses fewer parameters
* Performs well on small datasets
* Handles sequential text data effectively

---

# Future Improvements

* Larger dataset
* BiGRU implementation
* LSTM comparison
* Attention mechanism
* Streamlit web application
* Twitter API integration
* Transformer models like BERT

---

# Applications

This project can be used for:

* Social media sentiment analysis
* Product review analysis
* Customer feedback systems
* Brand monitoring
* Opinion mining

---

# Learning Outcomes

This project helps in learning:

* Natural Language Processing (NLP)
* Deep Learning
* GRU Neural Networks
* Text preprocessing
* Tokenization
* Multi-class classification
* TensorFlow/Keras workflow

---

# Author

Kavi IT

```
```
