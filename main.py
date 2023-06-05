import numpy as np
from flask import Flask, request,jsonify,render_template
import joblib
import pandas as pd
from flask_cors import CORS




#create flask app
app = Flask(__name__)
CORS(app)

# Load the joblib model
model = joblib.load('best_model.joblib')
model_health = joblib.load('svm_model.joblib')
model_env = joblib.load('env_model.joblib')
model_eco = joblib.load('eco_model.joblib')

@app.route('/', methods=['GET'])
def hello():
    return jsonify("hello world")

@app.route('/predict-poverty', methods=['POST'])
def predictPoverty():
    data = request.get_json()
    poverty_headcount = data['Poverty_headcount_at_2_15']
    poverty_gap = data['Poverty_gap_at_2_15']
    poverty_severity = data['Poverty_severity_2_15']
    X = pd.DataFrame({'Poverty_headcount_at_2_15': [poverty_headcount],
                      'Poverty_gap_at_2_15': [poverty_gap],
                      'Poverty_severity_2_15': [poverty_severity]})
    y_pred = model.predict(X)[0]
    result = {'Predicted_Poverty_rate': y_pred}
    return jsonify(result)
@app.route('/predict-health', methods=['POST'])
def predictHealth():
    data = request.get_json()
    DeathRate = data['DeathRate']
    IncidenceRate = data['IncidenceRate']
    PrevelanceRate = data['PrevelanceRate']
    ServiceRate = data['ServiceRate']
    X = pd.DataFrame({'DeathRate': [DeathRate],
                      'IncidenceRate': [IncidenceRate],
                      'PrevelanceRate': [PrevelanceRate],
                      'ServiceRate':[ServiceRate]})
    y_pred = model_health.predict(X)[0]
    result = {'Predicted_Disease_Condition': y_pred}
    return jsonify(result)
@app.route('/predict-eco', methods=['POST'])
def predictEco():
    data = request.get_json()
    Agriculture = data['Agriculture']
    Nonagriculture = data['Non_agriculture']
    Industry = data['Industry']
    Services = data['Services']
    Notclassified = data['Notclassified']

    X = pd.DataFrame({'Non-agriculture': [Nonagriculture],
                      'Agriculture':[Agriculture],
                      'Industry': [Industry],
                      'Services': [Services],
                      'Not classified':[Notclassified]})
    y_pred = model_eco.predict(X)[0]
    result = {'Predicted_Income_Level': y_pred}
    return jsonify(result)

@app.route('/predict-env', methods=['POST'])
def predictEnv():
    data = request.get_json()
    Agricultural_methane_emissions____of_total = data['Agricultural_methane_emissions____of_total']
    Agricultural_nitrous_oxide_emissions____of_total = data['Agricultural_nitrous_oxide_emissions____of_total']
    Methane_emissions____change_from_1990 = data['Methane_emissions____change_from_1990']
    Methane_emissions__kt_of_CO2_equivalent = data['Methane_emissions__kt_of_CO2_equivalent']
    Methane_emissions_in_energy_sector__thousand_metric_tons_of_CO2 = data['Methane_emissions_in_energy_sector__thousand_metric_tons_of_CO2']
    Nitrous_oxide_emissions____change_from_1990 = data['Nitrous_oxide_emissions____change_from_1990']
    Nitrous_oxide_emissions__thousand_metric_tons_of_CO2_equivalent = data['Nitrous_oxide_emissions__thousand_metric_tons_of_CO2_equivalent'],
    Nitrous_oxide_emissions_in_energy_sector____of_total = data['Nitrous_oxide_emissions_in_energy_sector____of_total'],
    Other_greenhouse_gas_emissions____change_from_1990 = data['Other_greenhouse_gas_emissions____change_from_1990'],
    Other_greenhouse_gas_emissions__HFC__PFC_and_SF6__thousand_metr = data['Other_greenhouse_gas_emissions__HFC__PFC_and_SF6__thousand_metr'],
    Total_greenhouse_gas_emissions____change_from_1990 = data['Total_greenhouse_gas_emissions____change_from_1990']
    X = pd.DataFrame({'Agricultural_methane_emissions____of_total': Agricultural_methane_emissions____of_total,
                      'Agricultural_nitrous_oxide_emissions____of_total': Agricultural_nitrous_oxide_emissions____of_total,
                      'Methane_emissions__kt_of_CO2_equivalent': Methane_emissions__kt_of_CO2_equivalent,
                      'Methane_emissions_in_energy_sector__thousand_metric_tons_of_CO2':Methane_emissions_in_energy_sector__thousand_metric_tons_of_CO2,
                      'Nitrous_oxide_emissions__thousand_metric_tons_of_CO2_equivalent': Nitrous_oxide_emissions__thousand_metric_tons_of_CO2_equivalent,
                      'Nitrous_oxide_emissions_in_energy_sector____of_total':Nitrous_oxide_emissions_in_energy_sector____of_total,
                      'Other_greenhouse_gas_emissions__HFC__PFC_and_SF6__thousand_metr':Other_greenhouse_gas_emissions__HFC__PFC_and_SF6__thousand_metr,
                      'Methane_emissions____change_from_1990':Methane_emissions____change_from_1990,
                      'Nitrous_oxide_emissions____change_from_1990':Nitrous_oxide_emissions____change_from_1990,
                      'Other_greenhouse_gas_emissions____change_from_1990':Other_greenhouse_gas_emissions____change_from_1990,
                      'Total_greenhouse_gas_emissions____change_from_1990':Total_greenhouse_gas_emissions____change_from_1990

                      })
    y_pred = model_env.predict(X)[0]
    result = {'Predicted_Pollution_Rate': y_pred}
    return jsonify(result)

if __name__ == '__main__':
    app.run()
