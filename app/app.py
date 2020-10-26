#!flask/bin/python
from flask import Flask, abort, jsonify, make_response, request
from housing.modeling import score as sr
from housing.preparation import utils as ut

score_cfg_path = "./flask_config.yml"
score_cfg = ut.read_config(score_cfg_path)
app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    observation = {
        "longitude": request.json.get("longitude", ""),
        "latitude": request.json.get("latitude", ""),
        "housing_median_age": request.json.get("housing_median_age", ""),
        "total_rooms": request.json.get("total_rooms", ""),
        "total_bedrooms": request.json.get("total_bedrooms", ""),
        "population": request.json.get("population", ""),
        "households": request.json.get("households", ""),
        "median_income": request.json.get("median_income", ""),
        "ocean_proximity": request.json.get("ocean_proximity", ""),
    }
    return jsonify({'prediction': sr.score(score_cfg, observation)[0]}), 201

# curl -i -H "Content-Type: application/json" -X POST -d '{"longitude": -120.430000, "latitude": 34.870000, "housing_median_age": 21.000000, "total_rooms": 2131.000000, "total_bedrooms": 329.000000, "population": 1094.000000, "households": 353.000000, "median_income": 4.664800, "ocean_proximity": "<1H OCEAN"}' http://localhost:5000/predict

if __name__ == '__main__':
    app.run(debug=True)
