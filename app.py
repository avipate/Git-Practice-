# Importing required Libraries
from flask import Flask, render_template

# Creating app
app = Flask(__name__)


# Creating home route
@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict')
def prediction_page():
    return render_template('prediction.html')


if __name__ == "__main__":
    app.run(port=3000, debug=True)