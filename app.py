import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("ML Model Prediction App")
st.write("Enter the values for prediction:")

# Let the user input features dynamically
num_features = st.number_input("How many features does your model expect?", min_value=1, value=3, step=1)

input_values = []
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}")
    input_values.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([input_values])
    try:
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
