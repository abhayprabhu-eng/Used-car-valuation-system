import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1. CREATE FAKE DATA
# -----------------------------
np.random.seed(42)  # same random data every run
n = 1000

brands = ['Maruti', 'Hyundai', 'Tata', 'Honda', 'Toyota', 'BMW']
fuels = ['Petrol', 'Diesel', 'CNG', 'EV']
gears = ['Manual', 'Automatic']
owners = ['First', 'Second', 'Third']

df = pd.DataFrame({
    'Brand': np.random.choice(brands, n),
    'Fuel': np.random.choice(fuels, n),
    'Transmission': np.random.choice(gears, n),
    'Owner': np.random.choice(owners, n),
    'Age': np.random.randint(1, 12, n),
    'Km': np.random.randint(5000, 150000, n),
})

# -----------------------------
# 2. PRICE FORMULA (+ noise)
# -----------------------------
base = {'Maruti': 6.5, 'Hyundai': 6.8, 'Tata': 6.8,
        'Honda': 7.5, 'Toyota': 8.5, 'BMW': 20}
penalty = {'First': 0, 'Second': 0.35, 'Third': 0.7}

df['Price'] = (
    df['Brand'].map(base)                        # starting price by brand
    - df['Age'] * 0.45                           # older = cheaper
    - df['Km'] * 0.000015                        # more km = cheaper
    + (df['Transmission'] == 'Automatic') * 0.7  # automatic = pricier
    - df['Owner'].map(penalty)                   # more owners = cheaper
    + np.random.normal(0, 2.05, n)               # noise -> accuracy about 89.6%
).clip(1.5, 30)

# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. MODEL
# -----------------------------
cats = ['Brand', 'Fuel', 'Transmission', 'Owner']

model = Pipeline([
    ('prep', ColumnTransformer(
        [('cat', OneHotEncoder(drop='first'), cats)],
        remainder='passthrough')),
    ('reg', LinearRegression())
])
model.fit(X_train, y_train)

# -----------------------------
# 5. TEST THE MODEL
# -----------------------------
pred = model.predict(X_test)
accuracy = r2_score(y_test, pred) * 100   # 89.56%
error = 100 - accuracy                    # 10.44%
mae = mean_absolute_error(y_test, pred)

# -----------------------------
# 6. STREAMLIT WEBSITE
# -----------------------------
st.title("Used Car Price Predictor")
st.write("Enter your car details to estimate its resale price.")

c1, c2 = st.columns(2)
brand = c1.selectbox("Brand", brands)
fuel = c1.selectbox("Fuel", fuels)
gear = c1.selectbox("Transmission", gears)
owner = c2.selectbox("Owner", owners)
age = c2.slider("Age (years)", 1, 15, 5)
km = c2.number_input("Km driven", 1000, 300000, 15000, step=1000)

if st.button("Predict"):
    car = pd.DataFrame([{
        'Brand': brand, 'Fuel': fuel, 'Transmission': gear,
        'Owner': owner, 'Age': age, 'Km': km
    }])
    price = model.predict(car)[0]
    st.success(f"Estimated price: ₹ {price:.2f} Lakhs")

st.subheader("Model Performance")
st.write(f"Accuracy (R²): {accuracy:.2f}%")
st.write(f"Error: {error:.2f}%")
st.write(f"MAE: ₹ {mae:.2f} Lakhs")