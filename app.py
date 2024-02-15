# -*- coding: utf-8 -*-
import json
import threading
from package.response import Response

from script.info_filter import *
from script.upload_csv_pipeline import *
from script.basic_setting import *
from script.training_pipeline import *
from script.prediction_pipeline import *

from flasgger import Swagger
from flask import Flask, request, send_file

setting()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ml_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SWAGGER'] = {
    "title": "auto-ml-api",
    "description": "是一款整合各列機器學習模型的機器學習預測策平台",
    "version": "0.0.0",
    "hide_top_bar": True
}

swagger = Swagger(app)
db.init_app(app)

with app.app_context():
    db.session.remove()
    db.drop_all()
    db.create_all()


@app.errorhandler(Exception)
def handle_exception(e: Exception):
    app.logger.error(e)
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
        description: Upload success.
        schema:
          type: object
          properties:
            description:
              type: string
              description: upload success.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The unique identifier for the uploaded file.
      400:
        description: Upload fail.
        schema:
          type: object
          properties:
            description:
              type: string
              description: upload fail.
            response:
              type: string
              description: Error message describing the reason for failure.
      500:
        description: Sever error.
        schema:
          type: object
          properties:
            description:
              type: string
              description: sever error
            response:
              type: string
              description: Error message describing the internal server error.
    """

    try:
        uploaded_file = request.files['file']
        project_name = request.form['project_name']
        assert uploaded_file.filename.endswith('.csv'), "upload file should be csv file formate"
    except Exception as e:
        return Response.client_error('upload fail', str(e))

    file_id = upload_csv_pipeline(uploaded_file, project_name)
    return Response.response('upload csv success', {"uuid": file_id})


@app.route('/api/csv_info', methods=['GET'])
def get_csv_info():
    """
    Get information about a CSV file.

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
        description: Success
        schema:
          type: object
          properties:
            file_name:
              type: string
              description: name of file
            project_name:
              type: string
              description: name of project
            row_num:
              type: integer
              description: length of row
            column_num:
              type: integer
              description: length of column
            columns:
              type: array
              items:
                type: string
            column_info:
              type: object
              properties:
                continuous_column:
                  type: object
                  properties:
                    count:
                      type: integer
                      description: length of this column
                    mean:
                      type: number
                      description: mean of this column
                    std:
                      type: number
                      description: std of this column
                    min:
                      type: number
                      description: min of this column
                    25%:
                      type: number
                      description: 25% of this column
                    50%:
                      type: number
                      description: 50% of this column
                    75%:
                      type: number
                      description: 75% of this column
                    max:
                      type: number
                      description: max of this column
                    column_class:
                      type: string
                      description: continuous variable
                    nan:
                      type: integer
                      description: null value of this column
                    total:
                      type: integer
                      description: length of this column
                discrete_column:
                  type: object
                  properties:
                    value_name:
                      type: integer
                      description: number of this value
                    column_class:
                      type: string
                      description: continuous variable
                    nan:
                      type: integer
                      description: null value of this column
                    total:
                      type: integer
                      description: length of this column
      404:
        description: File not found
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found
            response:
              type: string
              description: Error message describing the internal server error.
    """
    file_id = request.args.get('file_id')
    csv = CSV.query.filter_by(file_id=file_id).first()

    if not (csv and os.path.exists(csv.preprocessing_config_path)):
        return Response.not_found('file not found')

    with open(csv.preprocessing_config_path, 'r') as json_file:
        data = json.load(json_file)

    return Response.response('get csv info success', csv_info_filter(data))


@app.route('/api/all_csv', methods=['GET'])
def get_all_csv():
    """
    Get all CSV list.

    ---
    tags:
      - CSV

    responses:
      200:
        description: Get all csvs success
        schema:
          type: object
          properties:
            description:
              type: string
              description: get csvs success
            response:
              type: array
              items:
                type: object
                properties:
                  file_id:
                    type: string
                    description: uuid of csv file
                  project_name:
                    type: string
                    description: name of project
                  mission_type:
                    type: string
                    description: regression or classification
                  train_status:
                    type: string
                    description: pending or finish
                  predict_status:
                    type: string
                    description: pending or finish
    """
    csvs = DatabaseOperator.select_all(CSV)

    result = [{
        'file_id': csv.file_id,
        'project_name': csv.project_name,
        'file_name': csv.file_name,
        'mission_type': csv.mission_type,
        'train_status': csv.train_status,
    } for csv in csvs]

    return Response.response('get csvs success', result)


@app.route('/api/csv_corr', methods=['GET'])
def get_csv_corr():
    """
    Get the correlation matrix plot for a CSV file.

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
        description: Successful PNG retrieval.
        content:
          image/png:
            schema:
              type: string
              format: binary

      404:
        description: File not found
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found
            response:
              type: string
              description: Error message describing the internal server error.
    """
    file_id = request.args.get('file_id')
    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not (csv and os.path.exists(csv.correlation_matrix_path)):
        return Response.not_found('file not found')

    return send_file(csv.correlation_matrix_path)


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    Train machine learning models.

    ---
    tags:
      - AI

    parameters:
      - name: file_id
        in: formData
        type: string
        required: true
        description: The ID of the CSV file. (Passing csv uuid)

      - name: mission_type
        in: formData
        type: string
        required: true
        description: The type of machine learning mission.
        enum:
          - classification
          - regression

      - name: feature
        in: formData
        type: string
        required: true
        description: The feature column name. (Passing feature names, which split like csv by ',')

      - name: label
        in: formData
        type: string
        required: true
        description: The label column name. (Passing label name, which selected by radios)

    responses:
      200:
        description: Training success.
        schema:
          type: object
          properties:
            description:
              type: string
              description: training success.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The unique identifier for the uploaded file.
      400:
        description: Training fail.
        schema:
          type: object
          properties:
            description:
              type: string
              description: training fail.
            response:
              type: string
              description: Error message describing the reason for failure.
      404:
        description: File not Found.
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found.
            response:
              type: string
              description: Error message describing the reason for failure.
      500:
        description: Sever error.
        schema:
          type: object
          properties:
            description:
              type: string
              description: sever error
            response:
              type: string
              description: Error message describing the internal server error.
    """
    file_id = None
    mission_type = None
    feature = None
    label = None

    try:
        file_id = request.form['file_id']
        label = request.form['label']
        feature = request.form['feature'].split(',')
        mission_type = request.form['mission_type']
    except Exception as e:
        Response.client_error("input error", e)

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not (csv and os.path.exists(csv.processed_data_path)):
        return Response.not_found('file not found')

    threading.Thread(
        target=training_pipeline,
        args=(file_id, label, feature, mission_type, app), daemon=True
    ).start()

    return Response.response("training", {"uuid": file_id})


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
    file_id = request.args.get('file_id')

    csv = CSV.query.filter_by(file_id=file_id).first()
    if not (csv and os.path.exists(csv.cv_res_path)):
        return Response.not_found('file not found')

    return send_file(csv.cv_res_path)


@app.route('/api/train_pic', methods=['GET'])
def get_train_pic():
    """
    Get the training picture for a specific file.
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
        description: Successful PNG retrieval.
        content:
          image/png:
            schema:
              type: string
              format: binary

      404:
        description: File not found
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found
            response:
              type: string
              description: Error message describing the internal server error.
    """
    file_id = request.args.get('file_id')
    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})

    if not (csv and os.path.exists(csv.cv_res_path)):
        return Response.not_found('file not found')

    with open(csv.cv_res_path, 'r') as f:
        cv_result = json.load(f)

    queue = []
    if csv.mission_type == "classification":
        for k, v in cv_result['results'].items():
            queue.append((k, v['test_accuracy'] + v['test_average_precision'] + v['test_recall_weighted']))

    elif csv.mission_type['results'] == "regression":
        for k, v in cv_result.items():
            queue.append((k, -v['test_mean_absolute_error'] - v['test_mean_squared_error'] + v['test_r2']))

    model_name = sorted(queue, key=lambda x: x[1], reverse=True)[0][0]
    png_path = csv.raw_data_path.split('/')
    png_path[-1] = model_name + '.png'

    return send_file('/'.join(png_path))


@app.route('/api/predict_info', methods=['GET'])
def get_predict_info():
    """
    Get information about predict a CSV file.

    ---
    tags:
      - PREDICT

    parameters:
      - name: file_id
        in: query
        type: string
        required: true
        description: The ID of the CSV file.

    responses:
      200:
        description: Success
        schema:
          type: object
          properties:
            file_name:
              type: string
              description: name of file
            project_name:
              type: string
              description: name of project
            row_num:
              type: integer
              description: length of row
            column_num:
              type: integer
              description: length of column
            columns:
              type: array
              items:
                type: string
            column_info:
              type: object
              properties:
                continuous_column:
                  type: object
                  properties:
                    count:
                      type: integer
                      description: length of this column
                    mean:
                      type: number
                      description: mean of this column
                    std:
                      type: number
                      description: std of this column
                    min:
                      type: number
                      description: min of this column
                    25%:
                      type: number
                      description: 25% of this column
                    50%:
                      type: number
                      description: 50% of this column
                    75%:
                      type: number
                      description: 75% of this column
                    max:
                      type: number
                      description: max of this column
                    column_class:
                      type: string
                      description: continuous variable
                    nan:
                      type: integer
                      description: null value of this column
                    total:
                      type: integer
                      description: length of this column
                discrete_column:
                  type: object
                  properties:
                    value_name:
                      type: integer
                      description: number of this value
                    column_class:
                      type: string
                      description: continuous variable
                    nan:
                      type: integer
                      description: null value of this column
                    total:
                      type: integer
                      description: length of this column
      404:
        description: File not found
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found
            response:
              type: string
              description: Error message describing the internal server error.
    """
    file_id = request.args.get('file_id')
    csv = CSV.query.filter_by(file_id=file_id).first()

    if not (csv and os.path.exists(csv.cv_res_path)):
        return Response.not_found('file not found')

    with open(csv.preprocessing_config_path, 'r') as json_file:
        preprocessing_config = json.load(json_file)

    with open(csv.cv_res_path, 'r') as json_file:
        cv_res = json.load(json_file)

    predict_info = predict_info_filter(preprocessing_config, cv_res)
    return Response.response('get csv info success', predict_info)


@app.route('/api/predict', methods=['POST'])
def predict_csv():
    """
    Uploads a CSV file and saves information to the database.

    ---
    tags:
      - PREDICT

    parameters:
      - name: predict_payload
        in: formData
        type: string
        required: true
        description: The feature column name. (come from json dict)

    responses:
      200:
        description: Upload success.
        schema:
          type: object
          properties:
            description:
              type: string
              description: upload success.
            response:
              type: object
              properties:
                uuid:
                  type: string
                  description: The predict result.
      400:
        description: Upload fail.
        schema:
          type: object
          properties:
            description:
              type: string
              description: upload fail.
            response:
              type: string
              description: Error message describing the reason for failure.
      404:
        description: File not found
        schema:
          type: object
          properties:
            description:
              type: string
              description: file not found
            response:
              type: string
              description: Error message describing the internal server error.
      500:
        description: Sever error.
        schema:
          type: object
          properties:
            description:
              type: string
              description: sever error
            response:
              type: string
              description: Error message describing the internal server error.
    """

    predict_payload = request.form['predict_payload']
    predict_payload = json.loads(predict_payload)

    file_id = predict_payload['file_id']
    feature = predict_payload['feature']

    csv = DatabaseOperator.select_one(CSV, {'file_id': file_id})
    if not (csv and os.path.exists(csv.cv_res_path)):
        return Response.not_found('file not found')

    predict = prediction_pipeline(csv, feature)

    return Response.response('predict success', predict)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
