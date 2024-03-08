# -*- coding: utf-8 -*-
import json
import threading

import pandas as pd

from package.response import Response
from package.dataframe_operator import *
from package.ai_operator import *

from script.basic_setting import *
from script.upload_csv_pipeline import *
from script.train_model_pipeline import *

from flasgger import Swagger
from flask import Flask, request, send_file

import logging
import os.path
from logging.handlers import TimedRotatingFileHandler

setting()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ml_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SWAGGER'] = {
    "title": "auto-ml-api",
    "description": "是一款整合各列機器學習模型的機器學習預測策平台",
    "version": "0.1.0",
    "hide_top_bar": True
}

swagger = Swagger(app)
db.init_app(app)

# logging
if not os.path.exists('logging'):
    os.mkdir('logging')
handler = TimedRotatingFileHandler('logging/flask-error.log', when='midnight', interval=1, backupCount=7)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - IP: %(ip)s - Route: %(route)s - Message: %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

with app.app_context():
    db.session.remove()
    db.drop_all()
    db.create_all()

if not os.path.exists('file'):
    os.mkdir('file')


@app.errorhandler(Exception)
def handle_exception(e: Exception):
    ip = request.remote_addr if request else 'unknown'
    route = request.url_rule.rule if request.url_rule else 'unknown'
    app.logger.error(f"An exception occurred: {str(e)}", extra={'ip': ip, 'route': route})
    return Response.sever_error("sever error", str(e))


@app.route("/api/")
def test_connection():
    """
    Test API Connection.
    ---
    tags:
      - Testing
    responses:
      200:
        description: Connection successful.
    """
    return Response.response('connect success')


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Uploads a CSV file and saves information to the database.

    ---
    tags:
     - CSV

    parameters:
     - name: file
       in: formData
       type: file
       required: true
       description: The CSV file to upload.

     - name: project_name
       in: formData
       type: string
       required: true
       description: The name of the project associated with the CSV file.
    responses:
      200:
        description: Upload successful
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The unique identifier for the uploaded file.
      400:
        description: Client Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
      500:
        description: Server Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
    """
    if 'project_name' not in request.form:
        return Response.client_error('client error', "missing 'project_name' in request form")

    if 'file' not in request.files:
        return Response.client_error('client error', "missing 'file' in request files")

    file = request.files['file']
    project_name = request.form['project_name']

    if not file.filename.endswith('.csv'):
        return Response.client_error('client error', "upload file should be csv formate")

    file_id = upload_csv_pipeline(file, project_name)
    return Response.response('upload csv success', {"uuid": file_id})


@app.route('/api/all_csv', methods=['GET'])
def get_all_csv():
    """
    Uploads a CSV file and saves information to the database.

    ---
    tags:
     - CSV

    responses:
      200:
        description: Get CSV files success
        schema:
          type: object
          properties:
            description:
              type: string
              description: Get CSV files success.
            response:
              type: array
              items:
                type: object
                properties:
                  file_id:
                    type: string
                    description: The unique identifier for the uploaded file.
                  file_name:
                    type: string
                  project_name:
                    type: string
                  train_status:
                    type: string
    """
    csvs = DatabaseOperator.select_all(CSV)
    payload = [{
        'file_id': csv.file_id,
        'project_name': csv.project_name,
        'file_name': csv.file_name,
        'train_status': csv.train_status,
        'mission_type': csv.mission_type,
    } for csv in csvs]
    return Response.response('get csv success', payload)


@app.route('/api/csv_info', methods=['GET'])
def get_csv_info():
    """
    Uploads a CSV file and saves information to the database.

    ---
    tags:
     - CSV

    parameters:
      - name: file_id
        in: query
        type: string
        required: true
        description: The ID of the CSV file.

    responses:
      200:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: Get CSV info success.
            response:
              type: array
              items:
                type: object
    """
    if 'file_id' not in request.args:
        return Response.client_error('client error', "missing 'file_id' in request args")

    file_id = request.args.get('file_id')
    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})

    if not csv:
        return Response.not_found('file not found', f'{file_id=} not found')

    preprocessed_config = json.loads(csv.preprocessed_config)
    df = pd.read_csv(f'{csv.file_dir}/raw_data.csv')
    preprocessed_config['total_column'] = int(df.shape[1])
    preprocessed_config['total_row'] = int(df.shape[0])

    return Response.response('get csv success', preprocessed_config)


@app.route('/api/variable_type', methods=['POST'])
def change_variable_type():
    """
    Change variable type in csv file

    ---
    tags:
     - CSV

    parameters:
     - name: file_id
       in: formData
       type: string
       required: true
       description: The CSV file_id.

     - name: column
       in: formData
       type: string
       required: true
       description: Column name.

     - name: variable_type
       in: formData
       type: string
       required: true
       description: discrete variable or continuous variable.

    responses:
      200:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: change variable type success.
            response:
              type: string
      400:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: client error.
            response:
              type: string
      404:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found.
            response:
              type: string
      500:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: Get CSV info success.
            response:
              type: string
    """
    if 'file_id' not in request.form:
        return Response.client_error('client error', "missing 'file_id' in request form")

    if 'column' not in request.form:
        return Response.client_error('client error', "missing 'column' in request form")

    if 'variable_type' not in request.form:
        return Response.client_error('client error', "missing 'variable_type' in request form")

    file_id = request.form['file_id']
    column = request.form['column']
    variable_type = request.form['variable_type']

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})

    if not csv:
        return Response.not_found('file not found', f'{file_id=} not found')

    preprocessed_config = json.loads(csv.preprocessed_config)

    if column not in preprocessed_config:
        return Response.not_found('column not found', f'{column=} not found')
    if variable_type not in preprocessed_config[column]['allow_class']:
        return Response.not_found('variable type is not allow', f'{variable_type=} is not allow')
    preprocessed_config[column]['column_class'] = variable_type
    preprocessed_config[column]['na_processing'] = 'fill mode'

    csv.preprocessed_config = json.dumps(preprocessed_config)
    db.session.commit()

    return Response.response('change variable type success', preprocessed_config)


@app.route('/api/na_processing_type', methods=['POST'])
def change_na_processing_type():
    """
    Change variable type in csv file

    ---
    tags:
     - CSV

    parameters:
     - name: file_id
       in: formData
       type: string
       required: true
       description: The CSV file_id.

     - name: column
       in: formData
       type: string
       required: true
       description: Column name.

     - name: na_processing_type
       in: formData
       type: string
       required: true
       description: discrete variable or continuous variable.

    responses:
      200:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: change variable type success.
            response:
              type: string
      400:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: client error.
            response:
              type: string
      404:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found.
            response:
              type: string
      500:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: Get CSV info success.
            response:
              type: string
    """
    if 'file_id' not in request.form:
        return Response.client_error('client error', "missing 'file_id' in request form")

    if 'column' not in request.form:
        return Response.client_error('client error', "missing 'column' in request form")

    if 'na_processing_type' not in request.form:
        return Response.client_error('client error', "missing 'na_processing_type' in request form")

    file_id = request.form['file_id']
    column = request.form['column']
    na_processing_type = request.form['na_processing_type']

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})

    if not csv:
        return Response.not_found('file not found', f'{file_id=} not found')

    preprocessed_config = json.loads(csv.preprocessed_config)

    if column not in preprocessed_config:
        return Response.not_found('column not found', f'{column=} not found')
    if na_processing_type not in preprocessed_config[column]['allow_na_processing']:
        return Response.client_error('na processing type is not allow', f'{na_processing_type=} is not allow')
    if na_processing_type == 'fill unknown' and preprocessed_config[column]['column_class'] == 'continuous variable':
        return Response.client_error('continuous variable should not fill unknow', f'')

    preprocessed_config[column]['na_processing'] = na_processing_type
    csv.preprocessed_config = json.dumps(preprocessed_config)
    db.session.commit()

    return Response.response('change variable type success', preprocessed_config)


@app.route('/api/csv_corr', methods=['GET'])
def get_csv_corr():
    """
    Change variable type in csv file

    ---
    tags:
     - CSV

    parameters:
     - name: file_id
       in: query
       type: string
       required: true
       description: The CSV file_id.

    responses:
      200:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: change variable type success.
            response:
              type: string
      404:
        description: Get CSV info success
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found.
            response:
              type: string
    """
    if 'file_id' not in request.args:
        return Response.client_error('client error', "missing 'file_id' in request form")

    file_id = request.args.get('file_id')

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})

    if not csv:
        return Response.not_found('file not found', f'{file_id=} not found')
    if not os.path.exists(Path(csv.file_dir) / Path('raw_data.csv')):
        return Response.not_found('file not found', f'{file_id=} not found')

    dfo = DataFrameOperator(csv.file_dir, csv.preprocessed_config)
    dfo.preprocessing_train_csv()
    corr_path = dfo.save_correlation_matrix()

    return send_file(corr_path)


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    train AI model.

    ---
    tags:
     - AI

    parameters:
     - name: file_id
       in: formData
       type: string
       required: true
       description: The CSV file to upload.

     - name: feature
       in: formData
       type: string
       required: true
       description: The name of the project associated with the CSV file.

     - name: label
       in: formData
       type: string
       required: true
       description: The name of the project associated with the CSV file.

     - name: mission_type
       in: formData
       type: string
       required: true
       description: The name of the project associated with the CSV file.

    responses:
      200:
        description: Upload successful
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The unique identifier for the uploaded file.
      400:
        description: Client Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
      500:
        description: Server Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
    """
    if 'file_id' not in request.form or 'feature' not in request.form or 'label' not in request.form or 'mission_type' not in request.form:
        return Response.client_error('client error', "missing 'file_id or feature or label' in request form")

    file_id = request.form['file_id']
    feature = request.form['feature'].split(',')
    label = request.form['label']
    mission_type = request.form['mission_type']

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not csv:
        return Response.not_found('file not found')
    preprocessed_config = json.loads(csv.preprocessed_config)

    for column in feature + [label]:
        if column not in preprocessed_config:
            return Response.not_found('column not found', f"{column=} not found")
        if preprocessed_config[column]['column_class'] == 'drop variable':
            return Response.client_error('client error', f"{column=} not found")

    if preprocessed_config[label]['column_class'] == 'discrete variable':
        mission_type = 'classification'
        DatabaseOperator.update(CSV, {'file_id': file_id}, {'mission_type': mission_type})
    else:
        mission_type = 'regression'
        DatabaseOperator.update(CSV, {'file_id': file_id}, {'mission_type': mission_type})

    dfo = DataFrameOperator(csv.file_dir, csv.preprocessed_config)
    df = dfo.preprocessing_train_csv()

    threading.Thread(
        target=train_model_pipeline, args=(file_id, df, label, feature, mission_type, app), daemon=True
    ).start()

    return Response.response('train csv success', {"uuid": file_id})


@app.route('/api/train_result', methods=['GET'])
def get_train_result():
    """
        Get the training results for a specific file.
        ---
        tags:
          - AI
        parameters:
          - name: file_id
            in: query
            type: string
            required: true
            description: The unique ID of the file for which training results are requested.

        responses:
          200:
            description: Successful response with training results.
          404:
            description: File not found.
        """
    if 'file_id' not in request.args:
        return Response.client_error('client error', "missing 'file_id' in request args")

    file_id = request.args.get('file_id')

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not csv:
        return Response.not_found('file not found')

    return csv.train_result


@app.route('/api/prediction', methods=['POST'])
def get_prediction():
    """
    get AI prediction.

    ---
    tags:
     - AI

    parameters:
     - name: file_id
       in: formData
       type: string
       required: true
       description: The CSV file to upload.

     - name: model
       in: formData
       type: string
       required: true
       description: The model prediction used.

     - name: file
       in: formData
       type: file
       required: true
       description: The CSV file to upload.

    responses:
      200:
        description: Upload successful
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The unique identifier for the uploaded file.
      400:
        description: Client Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
      500:
        description: Server Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
    """
    if 'file_id' not in request.form:
        return Response.client_error('client error', "missing 'file_id' in request form")

    if 'model' not in request.form:
        return Response.client_error('client error', "missing 'model' in request files")

    if 'file' not in request.files:
        return Response.client_error('client error', "missing 'file' in request files")

    file_id = request.form['file_id']
    file = request.files['file']
    model = request.form['model']

    if model not in [
        'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
    ]:
        return Response.client_error('client error', f"{model=} is not valid")

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not csv:
        return Response.not_found('file not found')

    if csv.train_status != 'finished':
        return Response.sever_error('training is not finished')

    train_result = json.loads(csv.train_result)

    file.save(Path(f'file/{file_id}/predict_raw_data.csv'))

    dfo = DataFrameOperator(csv.file_dir, csv.preprocessed_config)
    df = dfo.preprocessing_predict_csv(feature=train_result['feature'])

    aio = AiOperator(file_id, df, train_result['label'], train_result['feature'], train_result['mission_type'], app)
    aio.get_predict()

    dfo.reverse_preprocessing_predict_csv(f'{csv.file_dir}/{model}_prediction.csv', feature=train_result['feature'])

    return send_file(f'{csv.file_dir}/{model}_prediction.csv')


@app.route('/api/train_pic', methods=['GET'])
def get_train_pic():
    """
    get AI prediction.

    ---
    tags:
     - AI

    parameters:
     - name: file_id
       in: query
       type: string
       required: true
       description: The CSV file to upload.

     - name: model
       in: query
       type: string
       required: true
       description: The model prediction used.

    responses:
      200:
        description: Upload successful
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The unique identifier for the uploaded file.
      400:
        description: Client Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
      500:
        description: Server Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
    """
    if 'file_id' not in request.args:
        return Response.client_error('client error', "missing 'file_id' in request form")

    if 'model' not in request.args:
        return Response.client_error('client error', "missing 'model' in request files")

    file_id = request.args['file_id']
    model = request.args['model']

    if model not in [
        'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
    ]:
        return Response.client_error('client error', f"{model=} is not valid")

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not csv:
        return Response.not_found('file not found')

    if not os.path.exists(Path(csv.file_dir) / Path(f'{model}.png')):
        return Response.sever_error('training is not finished')

    return send_file(Path(csv.file_dir) / Path(f'{model}.png'))


@app.route('/api/prediction_template', methods=['GET'])
def get_prediction_template():
    """
    get AI prediction csv template.

    ---
    tags:
     - AI

    parameters:
     - name: file_id
       in: query
       type: string
       required: true
       description: The CSV file to upload.

    responses:
      200:
        description: Upload successful

      400:
        description: Client Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
      500:
        description: Server Error
        schema:
          type: object
          properties:
            description:
              type: string
              description: Upload successful.
            response:
              type: string
              description: Error message.
    """
    if 'file_id' not in request.args:
        return Response.client_error('client error', "missing 'file_id' in request form")

    file_id = request.args['file_id']

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})

    if not csv:
        return Response.not_found('file not found')

    if csv.train_result == '':
        return Response.not_found('train not start')

    train_result = json.loads(csv.train_result)

    raw_data = pd.read_csv(Path(csv.file_dir) / Path('raw_data.csv'))

    payload = {}
    for f in train_result['feature']:
        payload[f] = raw_data[f].mode()[0]

    pd.DataFrame(payload, index=[0]).to_csv(Path(csv.file_dir) / Path('prediction_template.csv'), index=False)
    return send_file(Path(csv.file_dir) / Path('prediction_template.csv'), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
