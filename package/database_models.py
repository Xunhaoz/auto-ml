from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class CSV(db.Model):
    file_id = db.Column(db.String(36), primary_key=True, unique=True)
    file_name = db.Column(db.String(255))
    project_name = db.Column(db.String(255))
    mission_type = db.Column(db.String(255))
    train_status = db.Column(db.String(255))
    predict_status = db.Column(db.String(255))
    ori_file_path = db.Column(db.String(255))
    processed_file_path = db.Column(db.String(255))
    file_info_path = db.Column(db.String(255))
    file_pic_path = db.Column(db.String(255))
    file_cv_res_path = db.Column(db.String(255))
