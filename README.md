# Flipkart Review Sentiment Analysis with MLflow

This project builds a Machine Learning pipeline to classify Flipkart product reviews into Positive or Negative sentiments using Natural Language Processing (NLP). The experiment is tracked and managed using MLflow.

---

## Project Overview

Customer reviews are an important source of feedback for e-commerce platforms. This project analyzes review text and predicts the sentiment based on ratings.

- Ratings ≥ 4 → Positive
- Ratings ≤ 2 → Negative
- Neutral (Rating = 3) is removed

The model uses TF-IDF vectorization and Logistic Regression for classification.

---

## Objectives

- Perform sentiment classification on Flipkart reviews
- Track experiments using MLflow
- Log metrics, parameters, and artifacts
- Register trained model for reuse

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- MLflow
- Matplotlib

---

## Project Workflow

1. Load and clean dataset
2. Convert ratings to sentiment labels
3. Split data into training and testing sets
4. Create Pipeline:
   - TF-IDF Vectorizer
   - Logistic Regression Classifier
5. Train model
6. Evaluate using Accuracy and F1 Score
7. Log experiment details using MLflow
8. Save confusion matrix and register model

---

## MLflow Experiment Tracking

The following are tracked:

- Model Parameters (C, max_iter, max_features)
- Evaluation Metrics (Accuracy, F1 Score)
- Confusion Matrix Artifact
- Registered Model: `FlipkartSentimentClassifier`

---

## Project Structure

