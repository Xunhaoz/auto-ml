from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc

db = SQLAlchemy()


class Classification(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_id = db.Column(db.String(40))
    model = db.Column(db.String(40))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)


class Regression(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_id = db.Column(db.String(40))
    model = db.Column(db.String(40))
    mse = db.Column(db.Float)
    mae = db.Column(db.Float)
    r_square = db.Column(db.Float)


class CSV(db.Model):
    file_id = db.Column(db.String(36), primary_key=True, unique=True)
    file_name = db.Column(db.String(255))
    project_name = db.Column(db.String(255))
    mission_type = db.Column(db.String(255))
    status = db.Column(db.String(255))
    file_path = db.Column(db.String(255))
    file_info_path = db.Column(db.String(255))
    file_pic_path = db.Column(db.String(255))
