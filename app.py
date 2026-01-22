from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # These names must match the 'name' attributes in your HTML
        features = [
            float(request.form["radius"]),
            float(request.form["texture"]),
            float(request.form["perimeter"]),
            float(request.form["area"]),
            float(request.form["smoothness"])
        ]
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        prediction = "Benign" if pred == 1 else "Malignant"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


