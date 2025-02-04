import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        input_data = [float(x) for x in request.form.values()]
        input_array = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]

        # Convert prediction to species name
        species = ["Setosa", "Versicolor", "Virginica"]
        result = species[prediction]

        return render_template("index.html", prediction_text=f"Predicted Species: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port)
