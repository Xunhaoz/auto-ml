from package.ai_operator import *


def train_model_pipeline(file_id, df, label, feature, mission_type, app):
    DatabaseOperator.update(CSV, {'file_id': file_id}, {'train_status': 'pending'}, app)

    try:
        aio = AiOperator(file_id, df, label, feature, mission_type, app)
        aio.cross_validation()
        aio.plot_validation()
        aio.train_model()
    except Exception as e:
        DatabaseOperator.update(CSV, {'file_id': file_id}, {'train_status': 'failed'}, app)
        raise e
    DatabaseOperator.update(CSV, {'file_id': file_id}, {'train_status': 'finished'}, app)
