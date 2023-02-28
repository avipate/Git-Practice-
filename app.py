# Importing required Libraries
from flask import Flask, render_template

# Creating app
app = Flask(__name__)


# Creating home route
@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(port=3000, debug=True)