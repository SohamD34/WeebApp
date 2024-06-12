import uvicorn
import pandas as pd
from fastapi import FastAPI
from catboost import CatBoostClassifier

MODEL_PATH = "../model/cat_model.cbm" 

def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

def get_churn_probability(data, model):
    dataframe = pd.DataFrame.from_dict(data, orient='index').T
    churn_probability = model.predict_proba(dataframe)[0][1]
    return churn_probability


model = load_model()
app = FastAPI(title="Churn Prediction API", version="1.0")

@app.get('/')

def index():
    return {'message': 'CHURN Prediction API'}

@app.post('/predict/')

def predict_churn(data: dict):
    churn_probability = get_churn_probability(data, model)
    return {'Churn Probability': churn_probability}

if __name__ == '__main__':
    uvicorn.run("fast-api:app", host='127.0.0.1', port=5000)