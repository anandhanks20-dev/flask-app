import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Load Data
df = pd.read_csv('car_prediction_data.csv')

# 2. Feature Engineering
# We match EXACTLY what your app sends
df['Present_Price_log'] = np.log(df['Present_Price'] + 1)
df['Fuel_Type_Diesel'] = df['Fuel_Type'].apply(lambda x: 1 if x == 'Diesel' else 0)
df['Fuel_Type_Petrol'] = df['Fuel_Type'].apply(lambda x: 1 if x == 'Petrol' else 0)
df['Seller_Type_Individual'] = df['Seller_Type'].apply(lambda x: 1 if x == 'Individual' else 0)
df['Transmission_Manual'] = df['Transmission'].apply(lambda x: 1 if x == 'Manual' else 0)

# 3. Select Features
# We use the raw "Year" because Random Forest handles it fine
features = [
    'Year', 
    'Present_Price', 
    'Kms_Driven', 
    'Owner', 
    'Present_Price_log', 
    'Fuel_Type_Diesel', 
    'Fuel_Type_Petrol', 
    'Seller_Type_Individual', 
    'Transmission_Manual'
]

X = df[features]
y = df['Selling_Price']

# 4. Train Random Forest (This prevents negative predictions)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save
pickle.dump(model, open('car_price_model.pkl', 'wb'))
print("Success! Random Forest model created.")