import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("CarDekho_1000_Streamlit_88_90.csv")

# Remove Test Drive Car from Owner
df["Owner"] = df["Owner"].astype(str).str.strip()

df = df[
    df["Owner"].str.casefold() != "test drive car"
].copy()

# Input features
cols = [
    "Brand",
    "Fuel",
    "Transmission",
    "Owner",
    "Vehicle_Age",
    "Km_Driven"
]

# Separate features and target
X = pd.get_dummies(df[cols])
y = df["Selling_Price"] / 100000

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model prediction and evaluation
pred = model.predict(X_test)

r2 = r2_score(y_test, pred) * 100
mae = mean_absolute_error(y_test, pred)

# Streamlit application
st.title("Used Car Price Predictor")

st.write(
    "Enter your car details to estimate its resale price."
)

c1, c2 = st.columns(2)

brand = c1.selectbox(
    "Brand",
    sorted(df["Brand"].unique())
)

fuel = c1.selectbox(
    "Fuel",
    sorted(df["Fuel"].unique())
)

transmission = c1.selectbox(
    "Transmission",
    sorted(df["Transmission"].unique())
)

owner = c2.selectbox(
    "Owner",
    sorted(df["Owner"].unique())
)

age = c2.slider(
    "Age (years)",
    6, 12, 8
)

km = c2.number_input(
    "Km driven",
    1000,
    250000,
    15000,
    step=1000
)

# Predict resale price
if st.button("Predict"):

    car = pd.DataFrame(
        [[brand, fuel, transmission, owner, age, km]],
        columns=cols
    )

    car = pd.get_dummies(car)

    car = car.reindex(
        columns=X.columns,
        fill_value=0
    )

    prediction = model.predict(car)[0]

    st.success(
        f"Estimated price: ₹ {prediction:.2f} Lakhs"
    )

# Display model performance
st.subheader("Model Performance")

st.write(f"R² Score: {r2:.2f}%")

st.write(f"Mean Absolute Error: ₹ {mae:.2f} Lakhs")
