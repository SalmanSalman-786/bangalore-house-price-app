from flask import Flask, request, render_template
import joblib
import json
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Load column names
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]


# Prediction function
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)


# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        location = request.form["location"]
        sqft = float(request.form["sqft"])
        bath = int(request.form["bath"])
        bhk = int(request.form["bhk"])

        prediction = predict_price(location, sqft, bath, bhk)

    locations = data_columns[3:]  # first 3 are sqft, bath, bhk
    return render_template("index.html", locations=locations, prediction=prediction)


# Run server
if __name__ == "__main__":
    app.run(debug=True)
