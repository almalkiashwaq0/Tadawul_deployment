import streamlit as st
import requests
import json

# Title of the app
st.title("KMeans Clustering Prediction App")

# Function to send data to the FastAPI backend and get the prediction
def get_prediction_from_api(url, input_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(input_data))
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to get a response from the API"}

# Input fields for the user to enter features
opening = st.number_input("Opening Price", value=1.0)
top = st.number_input("Top Price", value=1.0)
lowest = st.number_input("Lowest Price", value=1.0)
closing = st.number_input("Closing Price", value=1.0)
change = st.number_input("Change", value=0.0)
change_percentage = st.number_input("Change Percentage", value=0.0)

# URL of the FastAPI backend (your Render deployment)
api_url = "https://tadawul-deployment.onrender.com/predict"

# Button to make a prediction
if st.button("Predict Cluster"):
    # Prepare the input data in the format expected by the API
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

