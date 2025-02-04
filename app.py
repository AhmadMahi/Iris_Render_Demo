from flask import Flask, request, render_template
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Train the logistic regression model on the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)
species = iris.target_names

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred = model.predict(features)[0]
            prediction = species[pred]
        except Exception as e:
            prediction = "Error in input values. Please check your inputs."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
