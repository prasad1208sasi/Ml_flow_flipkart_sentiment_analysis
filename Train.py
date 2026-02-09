import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# ---------------------------
# Load your dataset
# ---------------------------
df = pd.read_csv("data.csv")

# Keep only required columns
df = df[["Review text", "Ratings"]].dropna()

# Convert Ratings to Sentiment
# >=4 -> Positive(1), <=2 -> Negative(0), drop 3
df = df[df["Ratings"] != 3]
df["sentiment"] = (df["Ratings"] >= 4).astype(int)

X = df["Review text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# MLflow Experiment
# ---------------------------
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

C = 1.0
max_iter = 300

with mlflow.start_run(run_name="tfidf_logreg_baseline"):

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(C=C, max_iter=max_iter))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # ---------------------------
    # Log Parameters
    # ---------------------------
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("max_features", 5000)

    # ---------------------------
    # Log Metrics
    # ---------------------------
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # ---------------------------
    # Confusion Matrix Artifact
    # ---------------------------
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ---------------------------
    # Register Model
    # ---------------------------
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="FlipkartSentimentClassifier"
    )

    # ---------------------------
    # Tags
    # ---------------------------
    mlflow.set_tag("dataset", "flipkart_reviews")
    mlflow.set_tag("stage", "baseline")
    mlflow.set_tag("owner", "siva")

    print("Accuracy:", acc)
    print("F1:", f1)
