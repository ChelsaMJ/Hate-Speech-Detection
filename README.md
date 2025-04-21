# Hate Speech Detection using Deep Learning

## Overview

This project aims to build a deep learning-based system to automatically detect hate speech in social media text, specifically tweets. The model leverages Natural Language Processing (NLP) techniques for text preprocessing and employs a neural network model using Keras and TensorFlow for classification.

## Objectives

- Detect and classify tweets as hate speech, offensive language, or neither.
- Apply NLP techniques to clean and preprocess text data.
- Build and train a neural network model to achieve reliable prediction performance.

## Dataset

The dataset used in this project consists of labeled tweets with the following classes:
- `0` - Hate Speech
- `1` - Offensive Language
- `2` - Neither

The dataset was sourced from a CSV file and contains approximately 24,000 entries.

## Project Structure

1. **Data Loading and Exploration**  
   Load the dataset and examine basic statistics and class distributions.

2. **Text Preprocessing**  
   Clean the text by removing noise, stopwords, and applying lemmatization using NLTK.

3. **Tokenization and Padding**  
   Convert text into numerical sequences and pad them to ensure uniform input length.

4. **Model Building**  
   Construct a deep learning model using Keras Sequential API. Common layers include:
   - Embedding Layer
   - Bidirectional LSTM
   - Dense Layers

5. **Training and Evaluation**  
   Train the model on the preprocessed data and evaluate its performance using accuracy and loss metrics.

6. **Visualization**  
   Plot accuracy and loss curves to monitor model performance over epochs.

## Dependencies

- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- NLTK
- TensorFlow / Keras
- WordCloud
- scikit-learn
