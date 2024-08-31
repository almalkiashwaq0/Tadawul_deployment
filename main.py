from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"

# Load models
model = joblib.load("kmeans_8_model.joblib")
scaler_model = joblib.load("scaler_8_model.joblib")

# Define input feature class
class InputFeatures(BaseModel):
    Opening: float
    Top: float
    Lowest: float
    Closing: float
    Change: float
    ChangePercentage: float  # Changed 'Change %' to 'ChangePercentage' to avoid syntax issues

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Opening': input_features.Opening,
        'Top': input_features.Top,
        'Lowest': input_features.Lowest,
        'Closing': input_features.Closing,
        'Change': input_features.Change,
        'ChangePercentage': input_features.ChangePercentage,  # Use the modified key
    }

    # Convert dictionary to sorted feature list
    features_list = [dict_f[key] for key in sorted(dict_f)]

    # Scale features
    scaled_features = scaler_model.transform([features_list])

    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    # Preprocess input features
    data = preprocessing(input_features)

    # Predict using the model
    y_pred = model.predict(data)  # Use predict instead of fit_predict
    
    return {"pred": y_pred.tolist()[0]}
