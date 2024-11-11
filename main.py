import pickle
from fastapi import FastAPI
from typing import List
import logging
import joblib
import numpy as np
from text_preprocessor_package.preprocessor import TextPreprocessor

app = FastAPI()

MODEL_PATH = "mlartifacts/903674473291749414/c9e207bb277e419e8ea82472d6bf5ea7/artifacts/model/model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

@app.get("/healthcheck")  
def healthcheck():  
    return {"status": "healthy"}  

def get_model(model_path):
    return joblib.load(model_path)

def get_label_encoder(encoder_path):
    return joblib.load(encoder_path)

@app.post("/predict")
def predict(input_data: List[str]):
    
    model = get_model(MODEL_PATH)
    label_encoder = get_label_encoder(ENCODER_PATH)
 
    pred = model.predict(input_data)
    return "POSITIVE" if int(pred) == 1 else "NEGATIVE"