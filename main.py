import pickle
from fastapi import FastAPI
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from text_preprocessor_package.preprocessor import TextPreprocessor



app = FastAPI()
MODEL_PATH = "mlartifacts/903674473291749414/c9e207bb277e419e8ea82472d6bf5ea7/artifacts/model/model.pkl"
TFIDF_PATH = "model/tfidf_vectorizer.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

@app.get("/healthcheck")  
def healthcheck():  
    return {"status": "healthy"}  


def get_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)
    
def get_tfidf(tfidf_path):
    with open(tfidf_path, "rb") as f:
        return pickle.load(f)
    
def get_label_encoder(encoder_path):
    with open(encoder_path, "rb") as f:
        return pickle.load(f)

@app.post("/predict")
def predict(input_data: List[str]):
    
    model = get_model(MODEL_PATH)
    tfidf  = get_tfidf(TFIDF_PATH)
    label_encoder = get_label_encoder(ENCODER_PATH)
    
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', tfidf),
        ('clf', model)
    ])
    pred = pipeline.predict([input_data])
    return label_encoder.inverse_transform(pred)[0]
    
    
input_data = ["hate"]
model = get_model(MODEL_PATH)
pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('tfidf', TfidfVectorizer(max_features=10000, lowercase=True, stop_words='english')),
    ('clf', model)
])
predictions = pipeline.predict(input_data)[:, 1]
print(predictions)