import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load Data
df = pd.read_csv('car_prediction_data.csv')

# 2. Feature Engineering (Match the pkl structure exactly)
# We use log price as the model expects it
df['Present_Price_log'] = np.log(df['Present_Price'] + 1)

# 3. One-Hot Encoding (Manual approach to ensure exact column names)
# Fuel Type
df['Fuel_Type_Diesel'] = df['Fuel_Type'].apply(lambda x: 1 if x == 'Diesel' else 0)
df['Fuel_Type_Petrol'] = df['Fuel_Type'].apply(lambda x: 1 if x == 'Petrol' else 0)

# Seller Type
df['Seller_Type_Individual'] = df['Seller_Type'].apply(lambda x: 1 if x == 'Individual' else 0)

# Transmission
df['Transmission_Manual'] = df['Transmission'].apply(lambda x: 1 if x == 'Manual' else 0)

# 4. Select the EXACT 9 features expected by your app
# Order matters!
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

# 5. Train the Model (Linear Regression works best for this small dataset)
model = LinearRegression()
model.fit(X, y)

# 6. Save
pickle.dump(model, open('car_price_model.pkl', 'wb'))
print("Success! New car_price_model.pkl created.")
print(f"Model expects these {len(features)} features: {features}")