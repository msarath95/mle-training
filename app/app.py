#!flask/bin/python
from flask import (Flask, abort, jsonify, make_response, redirect,
                   render_template, request, session, url_for)
from flask_wtf import Form
from housing.modeling import score as sr
from housing.preparation import utils as ut
from wtforms.fields import FloatField, RadioField, SubmitField

score_cfg_path = "./flask_config.yml"
score_cfg = ut.read_config(score_cfg_path)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


class MedianHousingFeatures(Form):
    longitude = FloatField("Longitude")
    latitude = FloatField("Latitude")
    housing_median_age = FloatField("Housing Median Age")
    total_rooms = FloatField("Total Rooms")
    total_bedrooms = FloatField("Total Bedrooms")
    population = FloatField("Population")
    households = FloatField("Households")
    median_income = FloatField("Median Income")
    ocean_proximity = RadioField("Ocean Proximity", 
                                 choices=[("<1H OCEAN", "<1H OCEAN"), 
                                          ("<1H OCEAN", "<1H OCEAN"),
                                          ("INLAND", "INLAND"),
                                          ("ISLAND", "ISLAND"),
                                          ("NEAR BAY", "NEAR BAY"),
                                          ("NEAR OCEAN", "NEAR OCEAN")]),
    submit = SubmitField('Submit')


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def index():
    session['predict'] = 0
    return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if 'predict' not in session:
#         return redirect(url_for('index'))
#     form = MedianHousingFeatures()
#     if form.validate_on_submit():
#         return render_template('prediction.html', form=form)
#     return render_template('prediction.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
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
