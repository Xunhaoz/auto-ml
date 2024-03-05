import json
import os
import uuid
from pathlib import Path

from script.json_toolkit import *
from package.database_operator import *

import pandas as pd


class CsvOperator:
    def __init__(self, uploaded_file, project_name):
        self.uploaded_file, self.project_name = uploaded_file, project_name
        self.file_id = str(uuid.uuid4())
        self.file_dir = Path(f'file/{self.file_id}/')
        self.preprocessed_config = ""

    def save_csv(self):
        os.makedirs(self.file_dir)
        self.uploaded_file.save(self.file_dir / Path('raw_data.csv'))

    def gen_preprocessing_config(self):
        df = pd.read_csv(self.file_dir / Path('raw_data.csv'))

        preprocessing_config = {}

        for column in df.columns:
            column_series = df[column].copy()
            preprocessing_config[column] = {
                'column_class': '',
                'allow_class': [],
                'na_rate': 0,
                'na_processing': '',
                'allow_na_processing': [],
                'mode': None,
                'total': len(column_series),
                'discrete_column_info': {},
                'continuous_column_info': {},
            }

            preprocessing_config[column]['na_rate'] = self.round_2_float(column_series.isna().mean())

            # Determining if a column is a discrete variable or a continuous variable.
            excluded_from_pandas_describe = ['count']
            if column_series.dtype == 'object':
                preprocessing_config[column]['column_class'] = 'discrete variable'
                preprocessing_config[column]['allow_class'].append('discrete variable')

                preprocessing_config[column]['discrete_column_info'] = {
                    str(category): str(category_num) for category, category_num in column_series.value_counts().items()
                }
            elif len(column_series.value_counts()) < 20:
                preprocessing_config[column]['column_class'] = 'discrete variable'
                preprocessing_config[column]['allow_class'].extend(['discrete variable', 'continuous variable'])

                preprocessing_config[column]['discrete_column_info'] = {
                    str(category): int(category_num) for category, category_num in column_series.value_counts().items()
                }
                preprocessing_config[column]['continuous_column_info'] = {
                    str(describe_title): self.round_2_float(describe_value)
                    for describe_title, describe_value in column_series.describe().items()
                    if describe_title not in excluded_from_pandas_describe
                }
            else:
                preprocessing_config[column]['column_class'] = 'continuous variable'
                preprocessing_config[column]['allow_class'].append('continuous variable')
                preprocessing_config[column]['continuous_column_info'] = {
                    str(describe_title): self.round_2_float(describe_value)
                    for describe_title, describe_value in column_series.describe().items()
                    if describe_title not in excluded_from_pandas_describe
                }

            # Get the mode and convert it to Python native format.
            if isinstance(column_series.mode()[0], str):
                preprocessing_config[column]['mode'] = str(column_series.mode()[0])
            elif isinstance(column_series.mode()[0], np.integer):
                preprocessing_config[column]['mode'] = int(column_series.mode()[0])
            elif isinstance(column_series.mode()[0], np.float_):
                preprocessing_config[column]['mode'] = float(column_series.mode()[0])

            preprocessing_config[column]['na_processing'] = 'fill mode'
            preprocessing_config[column]['allow_na_processing'].append('fill mode')
            if 'discrete variable' in preprocessing_config[column]['allow_class']:
                preprocessing_config[column]['allow_na_processing'].append('fill unknown')

            if preprocessing_config[column]['na_rate'] > 0.4:
                preprocessing_config[column]['column_class'] = 'drop variable'
                preprocessing_config[column]['allow_class'].append('drop variable')

        self.preprocessed_config = json.dumps(preprocessing_config)

    def save_2_database(self):
        payload = {
            'file_id': self.file_id,
            'file_dir': self.file_dir.__str__(),
            'file_name': self.uploaded_file.filename,
            'project_name': self.project_name,
            'train_result': '',
            'mission_type': '',
            'train_status': 'not start',
            'preprocessed_config': self.preprocessed_config,
        }
        DatabaseOperator.insert(CSV, payload)

    @staticmethod
    def round_2_float(num, p=2):
        return round(float(num), p)
