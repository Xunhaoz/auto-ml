# -*- coding: utf-8 -*-

import os
import uuid
import subprocess
import json

from package.response import Response
from package.database_models import *
from package.dataframe_operator import DataframeOperator

from flasgger import Swagger
from flask import Flask, request, send_file

if not os.path.exists('file'):
    os.makedirs('file')

if not os.path.exists('./temp'):
    os.makedirs('./temp')

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
    Upload a CSV file.
    ---
    tags:
      - CSV
    parameters:
      - name: file
        in: formData
        type: file
        description: The CSV file to be uploaded.
    responses:
      200:
        description: Upload successful.
      400:
        description: Bad request.
      500:
        description: Internal server error.
    """
    file_id = str(uuid.uuid4())
    file_dir = f'file/{file_id}/'
    file_path = f'file/{file_id}/{file_id}.csv'
    info_path = f'file/{file_id}/old_{file_id}.json'
    pic_path = f'file/{file_id}/{file_id}.png'

    try:
        uploaded_file = request.files['file']
        assert uploaded_file.filename.endswith('.csv'), "upload file should be file formate"
    except Exception as e:
        return Response.client_error('upload fail', e)

    os.makedirs(file_dir)
    uploaded_file.save(file_path)
    dataframe_operator = DataframeOperator(file_path)
    dataframe_operator.save_info(info_path, uploaded_file.filename)
    dataframe_operator.save_pic(pic_path)

    db.session.add(CSV(
        file_id=file_id, file_name=uploaded_file.filename, file_path=file_path,
        file_info_path=info_path, file_pic_path=pic_path
    ))
    db.session.commit()

    return Response.response('upload success', {"uuid": file_id})


@app.route('/api/csv_info', methods=['GET'])
def get_csv_info():
    """
    Get information about a CSV file.
    This endpoint returns information about the specified CSV file.
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
      404:
        description: File not found
    """
    file_id = request.args.get('file_id')
    csv = CSV.query.filter_by(file_id=file_id).first()

    if not (csv and os.path.exists(csv.file_info_path)):
        return Response.not_found('file not found')

    return send_file(csv.file_info_path)


@app.route('/api/csv_corr', methods=['GET'])
def get_csv_corr():
    """
    Get the correlation matrix plot for a CSV file.
    This endpoint returns the correlation matrix plot image for the specified CSV file.
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
      404:
        description: File not found
    """
    file_id = request.args.get('file_id')
    csv = CSV.query.filter_by(file_id=file_id).first()

    if not (csv and os.path.exists(csv.file_pic_path)):
        return Response.not_found('file not found')

    return send_file(csv.file_pic_path)


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    Train machine learning models.

    This endpoint allows training machine learning models using a specified CSV file.

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
        description: Training started successfully.
      400:
        description: Client error.
      404:
        description: File not found.
    """
    file_id = None
    mission_type = None
    feature = None
    label = None

    try:
        file_id = request.form['file_id']
        mission_type = request.form['mission_type']
        feature = request.form['feature'].split(',')
        label = request.form['label']
    except Exception as e:
        Response.client_error("input error", e)

    csv = CSV.query.filter_by(file_id=file_id).first()
    if not (csv and os.path.exists(csv.file_path)):
        return Response.not_found('file not found')

    dataframe_operator = DataframeOperator(csv.file_path)
    mission_type = dataframe_operator.check_mission_type(mission_type)
    feature = dataframe_operator.check_feature(feature)
    label = dataframe_operator.check_label(label)

    models = ['xgboost', 'lightgbm', 'catboost']
    for model in models:
        subprocess.Popen(f"python ./script/train_{model}.py {csv.file_path} {label} {feature} {mission_type}",
                         shell=True)

    return Response.response("training", {"uuid": file_id})


@app.route('/api/train_progressing', methods=['GET'])
def get_train_progressing():
    """
    Get the progress of the training for a specific file.
    This endpoint retrieves the training progress for a given file identified by its unique ID.
    ---
    tags:
      - AI
    parameters:
      - name: file_id
        in: query
        type: string
        required: true
        description: The unique ID of the file for which the training progress is requested.

    responses:
      200:
        description: Successful response with training progress information.
      404:
        description: File not found.
    """
    file_id = request.args.get('file_id')

    finish = set()

    csvs = Classification.query.filter_by(file_id=file_id).all()
    for csv in csvs:
        finish.add(csv.model)

    csvs = Regression.query.filter_by(file_id=file_id).all()
    for csv in csvs:
        finish.add(csv.model)

    if len(finish) == 0:
        return Response.not_found("file not found")

    return Response.response(
        "get file progress", {'progressing': round(len(finish) / 3, 2), 'finish': bool(len(finish) == 3)})


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
    result = {}
    csvs = Classification.query.filter_by(file_id=file_id).all()
    for csv in csvs:
        result[csv.model] = {
            'accuracy': csv.accuracy,
            'precision': csv.precision,
            'recall': csv.recall
        }

    csvs = Regression.query.filter_by(file_id=file_id).all()
    for csv in csvs:
        result[csv.model] = {
            'mse': csv.mse,
            'mae': csv.mae,
            'r_square': csv.r_square
        }

    if len(result) == 0:
        return Response.not_found("file not found")

    return Response.response("get file result", result)

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
        description: Successful response with result picture.
      404:
        description: File not found.
    """
    file_id = request.args.get('file_id')
    result = []
    csvs = Classification.query.filter_by(file_id=file_id).order_by('accuracy').all()
    for csv in csvs:
        result.append(csv.model)

    csvs = Regression.query.filter_by(file_id=file_id).order_by('r_square').all()
    for csv in csvs:
        result.append(csv.model)

    if len(result) == 0:
        return Response.not_found("file not found")

    return send_file(f'file/{file_id}/{result[-1]}.png')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
