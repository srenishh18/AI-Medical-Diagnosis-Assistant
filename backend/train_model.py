import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def clean_symptom(symptom):
    if pd.isna(symptom):
        return None
    return str(symptom).strip().lower()

def train_and_save_model(dataset_path: str, model_save_path: str, symptoms_list_path: str):
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    y_raw = df['Disease']
    symptom_cols = df.columns[1:]
    
    print("Extracting unique symptoms...")
    unique_symptoms = set()
    for col in symptom_cols:
        for val in df[col]:
            cleaned = clean_symptom(val)
            if cleaned:
                unique_symptoms.add(cleaned)
                
    symptoms_list = sorted(list(unique_symptoms))
    print(f"Found {len(symptoms_list)} unique symptoms.")
    
    print("One-hot encoding dataset...")
    X = np.zeros((len(df), len(symptoms_list)), dtype=int)
    symptom_to_idx = {sym: idx for idx, sym in enumerate(symptoms_list)}
    
    for i in range(len(df)):
        for col in symptom_cols:
            val = clean_symptom(df.iloc[i][col])
            if val and val in symptom_to_idx:
                X[i, symptom_to_idx[val]] = 1
                
    y = np.array([str(loc).strip() for loc in y_raw])
    
    # Calculate symptom counts per disease for weighting logic in backend
    print("Calculating symptom counts per disease...")
    disease_symptom_counts = {}
    for d in np.unique(y):
        mask = (y == d)
        row_counts = np.sum(X[mask], axis=1)
        disease_symptom_counts[d] = int(np.max(row_counts))

    # Data Augmentation (Milder 1-symptom drop)
    print("Augmenting dataset...")
    X_augmented = [X]
    y_augmented = [y]
    np.random.seed(42)
    X_drop = X.copy()
    for i in range(len(X_drop)):
        symptom_indices = np.where(X_drop[i] == 1)[0]
        if len(symptom_indices) > 2:
            drop_idx = np.random.choice(symptom_indices, 1, replace=False)
            X_drop[i, drop_idx] = 0
            
    X_augmented.append(X_drop)
    X_final = np.vstack(X_augmented)
    y_final = np.concatenate([y, y])
    
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
    
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        oob_score=True,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("Performing Cross Validation...")
    cv_scores = cross_val_score(rf_model, X_final, y_final, cv=5)
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}%")
    
    rf_model.fit(X_train, y_train)
    
    print("Saving model and metadata...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf_model, model_save_path)
    joblib.dump(symptoms_list, symptoms_list_path)
    # Save the symptom counts for the match-ratio weighting logic
    joblib.dump(disease_symptom_counts, os.path.join(os.path.dirname(model_save_path), "disease_symptom_counts.pkl"))
    
    print("Training Complete!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "DiseaseAndSymptoms.csv")
    model_path = os.path.join(base_dir, "backend", "models", "rf_model.pkl")
    symptoms_path = os.path.join(base_dir, "backend", "models", "symptoms_list.pkl")
    train_and_save_model(dataset_path, model_path, symptoms_path)
