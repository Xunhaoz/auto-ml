import json
import pandas as pd
import xgboost, lightgbm, catboost
from sklearn.ensemble import VotingClassifier

from joblib import load


def prediction_pipeline(csv, feature):
    with open(csv.cv_res_path, 'r') as json_file:
        cv_res = json.load(json_file)

    with open(csv.preprocessing_config_path, 'r') as json_file:
        preprocessing_config = json.load(json_file)

    df = pd.DataFrame([[None if f == '' else f for f in feature]], columns=cv_res['feature'])

    for col in cv_res['feature']:
        rule_dict = preprocessing_config['column_info'][col]
        if rule_dict['nan_processing'] == 'fill Unknown':
            df[col].fillna('Unknown')
        elif rule_dict['nan_processing'] == 'fill mode':
            df[col].fillna(rule_dict['mode'])
        else:
            assert not df[col].isnull().any(), f'column {col} is required'

        if rule_dict['column_class'] == 'discrete variable':
            df[col] = df[col].map(rule_dict['encode_mapping'])
        elif rule_dict['column_class'] == 'continuous variable':
            df[col] = (float(df[col]) - rule_dict['mean']) / rule_dict['std']

    prediction_result = {}
    for model_name in cv_res['models']:
        model_path = csv.raw_data_path.split('/')
        model_path[-1] = model_name + '.joblib'
        model = load('/'.join(model_path))
        predict = model.predict(df)
        prediction_result[model_name] = predict

    result = {}
    rule_dict = preprocessing_config['column_info'][cv_res['label']]
    for model_name, prediction in prediction_result.items():
        prediction = prediction[0]
        if 'CatClassifier' in model_name:
            prediction = prediction[0]
        if rule_dict['column_class'] == 'discrete variable':
            result[model_name] = rule_dict['r_encode_mapping'][str(prediction)]
        elif rule_dict['column_class'] == 'continuous variable':
            result[model_name] = rule_dict['mean'] + rule_dict['std'] * prediction
            result[model_name] = round(float(result[model_name]), 4)
    return result
