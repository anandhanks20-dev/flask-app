import pickle
import numpy as np
import sklearn

print(f"Scikit-learn version: {sklearn.__version__}")

try:
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    print("\nModel Type:", type(model))
    
    # Try to find feature names in common sklearn locations
    if hasattr(model, 'feature_names_in_'):
        print("\nSUCCESS! The model expects these features in this order:")
        print(list(model.feature_names_in_))
    elif hasattr(model, 'n_features_in_'):
        print(f"\nThe model expects {model.n_features_in_} features (names not stored).")
    
    # Check if it's a pipeline and inspect the first step
    if hasattr(model, 'steps'):
        print("\nModel is a Pipeline.")
        first_step = model.steps[0][1]
        if hasattr(first_step, 'feature_names_in_'):
            print("Features expected by the first step:")
            print(list(first_step.feature_names_in_))

except Exception as e:
    print(f"\nError inspecting model: {e}")