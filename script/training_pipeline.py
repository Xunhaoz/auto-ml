from package.ai_operator import *


def training_pipeline(*args):
    file_id, label, feature, mission_type, app = args
    DatabaseOperator.update(
        CSV, {'file_id': file_id},
        {'train_status': 'pending', 'mission_type': mission_type}, app=app
    )

    try:
        ai_op = AiOperator(file_id, label, feature, mission_type, app)
        ai_op.fool_poof()
        ai_op.cross_validation()
        ai_op.plot_validation()
        ai_op.train_model()
    except Exception as e:
        DatabaseOperator.update(CSV, {'file_id': file_id}, {'train_status': 'failed'}, app=app)
        app.logger.error(e)
        return

    DatabaseOperator.update(CSV, {'file_id': file_id}, {'train_status': 'finished'}, app=app)
