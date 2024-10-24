import pickle
from fastapi import FastAPI
from typing import List


app = FastAPI()
MODEL_PATH = "mlartifacts/903674473291749414/c9e207bb277e419e8ea82472d6bf5ea7/artifacts/model/model.pkl"

# Health check endpoint
@app.get("/healthcheck")  
def healthcheck():  
    return {"status": "healthy"}  


def get_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)
    
def get_transformer():
    pass

@app.post("/predict")
def predict(input_data: List[str]):
    model = get_model(MODEL_PATH)
    predict_probas = model.predict_proba(input_data)[:, 1]
    return predict_probas.to_list()
    