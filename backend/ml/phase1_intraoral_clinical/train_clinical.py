import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_clinical_model():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data", "processed", "oral_cancer_prediction_dataset_clinical.csv"
    )
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    print("Loading clinical dataset...")
    df = pd.read_csv(data_path)

    # Features we want to use for risk prediction
    features = [
        "Age",
        "Gender",
        "Tobacco Use",
        "Alcohol Consumption",
        "Betel Quid Use",
        "HPV Infection",
        "Poor Oral Hygiene",
        "Family History of Cancer",
        "Oral Lesions",
        "Unexplained Bleeding",
        "White or Red Patches in Mouth"
    ]
    target = "Oral Cancer (Diagnosis)"

    df = df.dropna(subset=features + [target])

    X = df[features].copy()
    y = df[target].apply(lambda x: 1 if x == "Yes" else 0)

    # Encode categorical variables
    encoders = {}
    for col in X.columns:
        # Convert any non-numeric column or explicit object column to string first
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and encoders
    artifact_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "artifacts", "models"
    )
    os.makedirs(artifact_dir, exist_ok=True)
    
    model_path = os.path.join(artifact_dir, "clinical_rf_model.pkl")
    encoder_path = os.path.join(artifact_dir, "clinical_encoders.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(encoders, encoder_path)
    
    print(f"Model saved to {model_path}")
    print(f"Encoders saved to {encoder_path}")

if __name__ == "__main__":
    train_clinical_model()
