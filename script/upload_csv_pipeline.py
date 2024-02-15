from package.csv_operator import *
from package.database_operator import *


def upload_csv_pipeline(uploaded_file, project_name):
    csv_op = CsvOperator(uploaded_file, project_name)
    csv_op.save_csv()
    csv_op.preprocessing_data_and_save_preprocessing_config()
    csv_op.save_correlation_matrix()
    csv_op.save_2_database()
    return csv_op.file_id
