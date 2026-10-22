import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('used_car_price_model.pkl')

pipeline = load_model()

# Website User Interface Design
st.title("🚗 Used Car Price Prediction System")
st.write("Enter your vehicle specifications below to get an instant fair market price estimate.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox('Brand', ['Maruti', 'Hyundai', 'Tata', 'Mahonda', 'Honda', 'Toyota', 'BMW'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'EV'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner_type = st.selectbox('Owner Type', ['First', 'Second', 'Third'])

with col2:
    vehicle_age = st.slider('Vehicle Age (Years)', 1, 15, 3)
    kilometers_driven = st.number_input('Kilometers Driven', min_value=1000, max_value=300000, value=25000, step=1000)
    engine_displacement = st.selectbox('Engine Displacement (CC)', [1000, 1200, 1500, 2000, 2500])

# Predict Button Action
if st.button('Predict Resale Price'):
    input_data = pd.DataFrame([{
        'Brand': brand,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Owner_Type': owner_type,
        'Vehicle_Age': vehicle_age,
        'Kilometers_Driven': kilometers_driven,
        'Engine_Displacement': engine_displacement
    }])

    prediction = pipeline.predict(input_data)[0]
    final_price = round(prediction, 2)
    
    # Display Result nicely
    st.success(f"Estimated Market Resale Price: **₹ {final_price} Lakhs**")