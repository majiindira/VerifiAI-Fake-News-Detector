
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("Data/WELFake_Dataset.csv")

# 2. Clean dataset
df = df.dropna(subset=["label"]).copy()

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine title and article text
df["content"] = df["title"] + " " + df["text"]

df = df[df["content"].str.strip() != ""]

X = df["content"]
y = df["label"].astype(int)

print("Dataset loaded:", len(df), "articles")

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# 5. Train model
print("Training model... Please wait.")

model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 7. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_news_model.joblib")

# Save evaluation results
metrics = {
    "accuracy": round(float(accuracy), 4),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open("models/evaluation_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("Location: models/fake_news_model.joblib")
