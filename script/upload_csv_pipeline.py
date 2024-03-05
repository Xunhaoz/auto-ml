from package.csv_operator import *
from package.database_operator import *


def upload_csv_pipeline(file, project_name):
    csv_op = CsvOperator(file, project_name)
    csv_op.save_csv()
    csv_op.gen_preprocessing_config()
    csv_op.save_2_database()
    return csv_op.file_id
