import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Train model automatically on startup to avoid version/pkl errors
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1200
    brands = ['Maruti', 'Hyundai', 'Tata', 'Mahonda', 'Honda', 'Toyota', 'BMW']
    fuel_types = ['Petrol', 'Diesel', 'CNG', 'EV']
    transmissions = ['Manual', 'Automatic']
    owner_types = ['First', 'Second', 'Third']

    data = {
        'Brand': np.random.choice(brands, n_samples, p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.05, 0.05]),
        'Fuel_Type': np.random.choice(fuel_types, n_samples, p=[0.50, 0.35, 0.10, 0.05]),
        'Transmission': np.random.choice(transmissions, n_samples, p=[0.75, 0.25]),
        'Owner_Type': np.random.choice(owner_types, n_samples, p=[0.70, 0.20, 0.10]),
        'Vehicle_Age': np.random.randint(1, 12, n_samples),
        'Kilometers_Driven': np.random.randint(5000, 150000, n_samples),
        'Engine_Displacement': np.random.choice([1000, 1200, 1500, 2000, 2500], n_samples, p=[0.3, 0.4, 0.2, 0.08, 0.02])
    }

    df = pd.DataFrame(data)

    base_price = 10.0
    df['Resale_Price'] = (
        base_price
        - (df['Vehicle_Age'] * 0.7)
        - (df['Kilometers_Driven'] * 0.00003)
        + (df['Engine_Displacement'] * 0.002)
        + np.where(df['Transmission'] == 'Automatic', 1.5, 0)
        + np.where(df['Brand'] == 'Toyota', 2.0, 0)
        + np.where(df['Brand'] == 'BMW', 8.0, 0)
        + np.random.normal(0, 0.8, n_samples)
    )
    df['Resale_Price'] = np.round(df['Resale_Price'].clip(1.5, 45.0), 2)

    X = df.drop(columns=['Resale_Price'])
    y = df['Resale_Price']

    categorical_features = ['Brand', 'Fuel_Type', 'Transmission', 'Owner_Type']
    numerical_features = ['Vehicle_Age', 'Kilometers_Driven', 'Engine_Displacement']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline

pipeline = train_model()

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
    
    st.success(f"Estimated Market Resale Price: **₹ {final_price} Lakhs**")