from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class CSV(db.Model):
    file_id = db.Column(db.String(36), primary_key=True, unique=True)
    file_name = db.Column(db.String(255))
    project_name = db.Column(db.String(255))
    mission_type = db.Column(db.String(255))
    train_status = db.Column(db.String(255))
    raw_data_path = db.Column(db.String(255))
    processed_data_path = db.Column(db.String(255))
    preprocessing_config_path = db.Column(db.String(255))
    correlation_matrix_path = db.Column(db.String(255))
    cv_res_path = db.Column(db.String(255))
