import joblib
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load models
crop_model = joblib.load(open('crop_app', 'rb'))
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('index2.html')

@app.route('/form', methods=["POST"])
def brain():
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])

    values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]

    if 0 < Ph <= 14 and Temperature < 100 and Humidity > 0:
        arr = [values]
        acc = crop_model.predict(arr)
        return render_template('prediction.html', prediction=str(acc))
    else:
        return "Sorry... Error in entered values in the form. Please check the values and fill it again."

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index2.html', prediction=prediction[0][0])

if __name__ == '__main__':
    app.run(debug=True)
