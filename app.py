from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)  

# Home Page
@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")


# Train Route
@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"


# Prediction Route
@app.route('/predict', methods=['POST', 'GET'])
def predict_route():
    if request.method == 'POST':
        try:
            data = [
                float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['citric_acid']),
                float(request.form['residual_sugar']),
                float(request.form['chlorides']),
                float(request.form['free_sulfur_dioxide']),
                float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                float(request.form['pH']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])
            ]

            data = np.array(data).reshape(1, 11)

            obj = PredictionPipeline()
            prediction = obj.predict(data)

            return render_template('results.html', prediction=str(prediction))

        except Exception as e:
            return f"Error Occurred: {e}"

    return render_template('index.html')


