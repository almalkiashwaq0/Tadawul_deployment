from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return "Welcome To Stock Data Clustering Project"


model = joblib.load('dbscan_model.joblib')
scaler_model = joblib.load('robust1_scaler.joblib')

class InputFeatures(BaseModel):
    Opening: float
    Top: float
    Lowest: float
    Closing: float
    Change: float
    ChangePercentage: float 

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Opening': input_features.Opening,
        'Top': input_features.Top,
        'Lowest': input_features.Lowest,
        'Closing': input_features.Closing,
        'Change': input_features.Change,
        'ChangePercentage': input_features.ChangePercentage, 
    }

    
    features_list = [dict_f[key] for key in sorted(dict_f)]

    scaled_features = scaler_model.transform([features_list])

    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    
    data = preprocessing(input_features)

    y_pred = model.fit_predict(data) 
    
    return {"pred": y_pred.tolist()[0]}
