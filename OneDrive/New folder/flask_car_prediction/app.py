from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import datetime
import os

app = Flask(__name__)

# Ensure paths are correct regardless of how the script is run
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'car_price_model.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'car_prediction_data.csv')

# Load trained model
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Error: car_price_model.pkl not found.")

# Load CSV for dropdowns with robust string handling
try:
    data = pd.read_csv(CSV_PATH)
    
    # Helper function to clean string columns
    def clean_column(col_name):
        if col_name in data.columns:
             # Convert to string, strip whitespace, get unique, sort
            return sorted(data[col_name].astype(str).str.strip().unique())
        return []

    car_names = clean_column("Car_Name")
    fuel_types = clean_column("Fuel_Type")
    seller_types = clean_column("Seller_Type")
    # Specifically fixing transmission by ensuring it's treated as clean strings
    transmissions = clean_column("Transmission") 
    owners = sorted(data["Owner"].unique())
    years = sorted(data["Year"].unique(), reverse=True)
    print("CSV data loaded and cleaned successfully.")

except FileNotFoundError:
    print("Error: car_prediction_data.csv not found.")
    car_names, fuel_types, seller_types, transmissions, owners, years = [], [], [], [], [], []
except Exception as e:
    print(f"An error occurred loading CSV: {e}")
    car_names, fuel_types, seller_types, transmissions, owners, years = [], [], [], [], [], []


@app.route("/")
def home():
    return render_template(
        "index.html",
        car_names=car_names,
        fuel_types=fuel_types,
        seller_types=seller_types,
        transmissions=transmissions,
        owners=owners,
        years=years
    )

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template("index.html", prediction_text="Error: Model file missing.")

    try:
        # 1. Grab inputs
        year_purchased = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        fuel_type = request.form["fuel_type"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]
        owner = int(request.form["owner"])

        # 2. Calculate Age
        current_year = datetime.datetime.now().year
        years_old = current_year - year_purchased

        # 3. Feature Engineering
        present_price_log = np.log(present_price + 1)
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_petrol = 1 if fuel_type == "Petrol" else 0
        seller_individual = 1 if seller_type == "Individual" else 0
        transmission_manual = 1 if transmission == "Manual" else 0

        # 4. Create Feature Array (Using Age instead of Year)
        final_features = np.array([[years_old, present_price, kms_driven, owner, 
                                    present_price_log,
                                    fuel_diesel, fuel_petrol,
                                    seller_individual,
                                    transmission_manual]])

        prediction = model.predict(final_features)[0]
        
        output = round(prediction, 2)
        if output < 0: output = 0
        result_text = f"₹ {output} Lakhs"

    except Exception as e:
        result_text = f"Error during calculation: {e}"
        # Default values if calculation fails so form doesn't crash
        year_purchased, present_price, kms_driven, owner = 0,0,0,0
        fuel_type, seller_type, transmission = "","",""

    return render_template(
        "index.html",
        prediction_text=result_text,
        # Pass necessary data back for dropdowns
        years=years, fuel_types=fuel_types, seller_types=seller_types, 
        transmissions=transmissions, owners=owners,
        # Pass back selected values to retain form state
        selected_year=year_purchased, selected_price=present_price,
        selected_kms=kms_driven, selected_fuel=fuel_type, 
        selected_seller=seller_type, selected_transmission=transmission, 
        selected_owner=owner
    )

if __name__ == "__main__":
    # debug=True allows you to see errors in the browser
    app.run(debug=True)