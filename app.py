import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("CarDekho_1000_Streamlit_88_90.csv")

cols = ['Brand', 'Fuel', 'Transmission', 'Owner', 'Vehicle_Age', 'Km_Driven']

X = pd.get_dummies(df[cols])
y = df['Selling_Price'] / 100000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred) * 100
mae = mean_absolute_error(y_test, pred)

st.title("Used Car Price Predictor")
st.write("Enter your car details to estimate its resale price.")

c1, c2 = st.columns(2)
brand = c1.selectbox("Brand", sorted(df['Brand'].unique()))
fuel = c1.selectbox("Fuel", sorted(df['Fuel'].unique()))
transmission = c1.selectbox("Transmission", sorted(df['Transmission'].unique()))
owner = c2.selectbox("Owner", sorted(df['Owner'].unique()))
age = c2.slider("Age (years)", 6, 12, 8)
km = c2.number_input("Km driven", 1000, 250000, 15000, step=1000)

if st.button("Predict"):
    car = pd.DataFrame([[brand, fuel, transmission, owner, age, km]], columns=cols)
    car = pd.get_dummies(car).reindex(columns=X.columns, fill_value=0)
    st.success(f"Estimated price: ₹ {model.predict(car)[0]:.2f} Lakhs")

st.subheader("Model Performance")
st.write(f"Accuracy (R²): {r2:.2f}%")
st.write(f"Error (MAE): ₹ {mae:.2f} Lakhs")