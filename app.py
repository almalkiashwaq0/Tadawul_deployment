import streamlit as st
import requests
import json

# Title of the app
st.title("Stock Data Clustering Project")

# Function to send data to the FastAPI backend and get the prediction
def get_prediction_from_api(url, input_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(input_data))
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to get a response from the API"}

# Input fields for the user to enter features
opening = st.slider("Opening Price", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
top = st.slider("Top Price", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
lowest = st.slider("Lowest Price", min_value=0.0, max_value=700.0, value=1.0, step=0.1)
closing = st.slider("Closing Price", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
change = st.slider("Change", min_value=-200.0, max_value=200.0, value=0.0, step=0.1)
change_percentage = st.slider("Change Percentage", min_value=-100.0, max_value=200.0, value=0.0, step=0.1)


api_url = "https://tadawul-deployment.onrender.com/predict"

# Button to make a prediction
if st.button("Predict Cluster"):
    input_data = {
        "Opening": opening,
        "Top": top,
        "Lowest": lowest,
        "Closing": closing,
        "Change": change,
        "ChangePercentage": change_percentage
    }
    
    # Get the prediction from the FastAPI backend
    result = get_prediction_from_api(api_url, input_data)
    
    # Display the result
    if "pred" in result:
        st.success(f"The data point belongs to cluster: {result['pred']}")
    else:
        st.error(f"Error: {result.get('error', 'Unknown error')}")

