import pandas as pd
import joblib

# Load your trained model
model = joblib.load('diabetes_model.pkl')

def predict_diabetes(features):
    """Predict the probability of diabetes based on input features."""
    # Corrected typo in 'columns'
    features = pd.DataFrame([features], columns=['Glucose', 'Blood_Pressure', 'Age', 'BMI', 'Skin_Thickness', 'Insulin', 'Pregnancies'])
    probability = model.predict_proba(features)[0][1]  # Probability of class '1' (diabetes)
    return probability
