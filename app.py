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
    response = {}
    response["winner"] = gomoku.Gomoku(data["board"]).check_winner()
    return json.dumps(response)

@app.route('/api/getnextmove/', methods=['POST'])
def get_next_move():
    data = request.json
    (x,y) = gomoku.Gomoku(data["board"]).get_next_move(data["cur"])
    response = {}
    response["x"]=x
    response["y"]=y
    return json.dumps(response)


if __name__ == "__main__":
    gomoku.precompute_gobal_sub_row()
    app.run()
