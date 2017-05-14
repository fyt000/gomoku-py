from flask import Flask
from flask import request
from flask_cors import CORS
import math
import json
import gomoku

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return "please make GET request to /api/iswinner/"


@app.route('/api/iswinner/', methods=['GET', 'POST'])
def iswinner():
    data = request.json
    print(data)
    response = {}
    response["winner"] = gomoku.check_winner(data["board"])
    return json.dumps(response)


if __name__ == "__main__":
    app.run()
