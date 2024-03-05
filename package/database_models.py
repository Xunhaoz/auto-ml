from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String, JSON, Column

db = SQLAlchemy()


class CSV(db.Model):
    file_id = Column(String(36), primary_key=True, unique=True)
    file_dir = Column(String(255))
    file_name = Column(String(255))
    project_name = Column(String(255))
    train_status = Column(String(40))
    mission_type = Column(String(20))
    train_result = Column(JSON)
    preprocessed_config = Column(JSON)
