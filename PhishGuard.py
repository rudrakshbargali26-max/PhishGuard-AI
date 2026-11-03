import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Phishing Detection using ML

print("=" * 60)
print("             PHISHGUARD-AI")
print("       Phishing Detection System")
print("=" * 60)


# 1. LOAD DATASET
print("\nLoading dataset...")

df = pd.read_csv("dataset_B_05_2020.csv")

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# 2. CHECK TARGET CLASSES

print("\nClass distribution:")

print(df["status"].value_counts())


# 3. PREPARE FEATURES

# The URL itself is text.
# For this first baseline, we use the numerical
# phishing-related features provided in the dataset.

X = df.drop(columns=["url", "status"])

y = df["status"]


# 4. HANDLE MISSING VALUES


print("\nChecking missing values...")

missing_values = X.isnull().sum().sum()

print("Total missing values:", missing_values)

# Replace missing values with 0
X = X.fillna(0)



# 5. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 6. TRAIN RANDOM FOREST MODEL

print("\nTraining PhishGuard-AI model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")



# 7. MAKE PREDICTIONS

y_pred = model.predict(X_test)



# 8. MODEL PERFORMANCE

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("              MODEL RESULTS")
print("=" * 60)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ]
    )
)


# 9. CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)



# 10. TEST A REAL URL FROM THE DATASET

print("\n" + "=" * 60)
print("             SAMPLE PREDICTION")
print("=" * 60)

sample_index = 0

sample_url = df.iloc[sample_index]["url"]

sample_features = df.iloc[
    sample_index
].drop(
    labels=["url", "status"]
)

sample_features = sample_features.fillna(0)

sample_features = sample_features.values.reshape(
    1, -1
)

prediction = model.predict(
    sample_features
)[0]

probability = model.predict_proba(
    sample_features
)[0]


print("\nURL:")
print(sample_url)


if prediction == 1:
    print("\nPrediction: PHISHING 🚨")
else:
    print("\nPrediction: LEGITIMATE ✅")


print(
    "Phishing probability:",
    round(probability[1] * 100, 2),
    "%"
)

print(
    "Legitimate probability:",
    round(probability[0] * 100, 2),
    "%"
)


print("\n" + "=" * 60)
print("          PHISHGUARD-AI FINISHED")
print("=" * 60)