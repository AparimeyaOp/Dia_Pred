from flask import Flask, render_template, request
import joblib
from prob import predict_diabetes

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_dia.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('Age')
    glucose = request.form.get('Glucose')
    bloodpressure = request.form.get('Blood Pressure')
    bmi = request.form.get('BMI')
    pregnancies = request.form.get('Pregnancies')  # Fixed spelling
    insulin = request.form.get('Insulin')
    skin_thickness = request.form.get('Skin Thickness')

    # Make sure to convert inputs to the appropriate types
    features = [int(age), float(glucose), float(bloodpressure), float(bmi), int(pregnancies), float(skin_thickness), float(insulin)]
    probability = predict_diabetes(features)

    return render_template('result.html', probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
