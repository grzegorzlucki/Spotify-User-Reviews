import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import mlflow
import optuna
from mlflow.models import infer_signature
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from text_preprocessor_package.preprocessor import TextPreprocessor


data = pd.read_csv('./data/DATASET.CSV')
data.dropna(subset=['Review'], inplace=True)

label_encoder = LabelEncoder()
labels = data['label'].values
X = data['Review'].values

y = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("spotify_classification")

def evaluation(trial):
    with mlflow.start_run(run_name=f"trial-{trial.number}") as run:
        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)

        mlflow.log_params(params)

        pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('tfidf', TfidfVectorizer(max_features=10000, lowercase=True, stop_words='english')),
            ('clf', SVC(**params, random_state=42))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric('f1_score', f1)

        return f1

study = optuna.create_study(direction='maximize')
study.optimize(evaluation, n_trials=10)

best_trial = study.best_trial
print(f"Best trial number: {best_trial.number}")
print(f"Best trial value (f1_score): {best_trial.value}")

with mlflow.start_run(run_name="best_model"):
    best_params = best_trial.params

    best_pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=10000, lowercase=True, stop_words='english')),
        ('clf', SVC(**best_params, random_state=42))
    ])

    best_pipeline.fit(X_train, y_train)
    
    signature = infer_signature(X_train, best_pipeline.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train
    )
    
    mlflow.log_params(best_params)
    
    y_pred = best_pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_score", f1)

print("Best model has been logged to MLflow.")

test_string = "Spotify is constantly logging me out of my account. Very frustrating experience!"
pred = best_pipeline.predict([test_string])
print(f"Prediction for test string: {label_encoder.inverse_transform(pred)[0]}")