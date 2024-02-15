import os
import uuid
from script.json_toolkit import *
from package.database_operator import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class CsvOperator:
    def __init__(self, uploaded_file, project_name):
        self.uploaded_file, self.project_name = uploaded_file, project_name
        self.file_id = str(uuid.uuid4())
        self.file_dir = f'file/{self.file_id}/'
        self.raw_data_path = f'{self.file_dir}raw_data.csv'
        self.processed_data_path = f'{self.file_dir}processed_data.csv'
        self.preprocessing_config_path = f'{self.file_dir}preprocessing_config.json'
        self.correlation_matrix_path = f'{self.file_dir}correlation_matrix.png'
        self.cv_res_path = f'{self.file_dir}cv_res.json'

    def save_csv(self):
        os.makedirs(self.file_dir)
        self.uploaded_file.save(self.raw_data_path)

    def preprocessing_data_and_save_preprocessing_config(self):
        df = pd.read_csv(self.raw_data_path)

        preprocessing_config = {
            'project_name': self.project_name,
            'file_name': self.uploaded_file.filename,
            'row_num': df.shape[0],
            'column_num': df.shape[1],
            'columns': df.columns
        }

        column_info = {}
        for column in df.columns:
            column_info[column] = {
                'column_class': self.is_discrete_continuous(df[column]),
                'total': df.shape[0],
                'nan': int(df[column].isnull().sum()),
                'required': not df[column].isnull().any(),
                'mode': df[column].mode().iloc[0]
            }

            # 離散變量
            if self.is_discrete_continuous(df[column]) == 'discrete variable':
                for k, v in df[column].value_counts().items():
                    column_info[column][str(k)] = v

                # 缺失值非常多
                if df[column].isnull().mean() > 0.2:
                    df = df.drop(columns=[column])
                    column_info[column]['nan_processing'] = 'drop'
                    continue

                # 缺失值中等
                elif 0.2 >= df[column].isnull().mean() > 0.1:
                    df[column] = df[column].fillna('Unknown')
                    column_info[column]['nan_processing'] = 'fill Unknown'

                # 缺失值不多
                elif 0.1 > df[column].isnull().mean() > 0:
                    df[column] = df[column].fillna(df[column].mode().iloc[0])
                    column_info[column]['nan_processing'] = 'fill mode'

                else:
                    column_info[column]['nan_processing'] = 'no operation'

                oe = OrdinalEncoder()
                df[column] = oe.fit_transform(df[[column]])
                df[column] = df[column].astype(int)
                column_info[column]['encode_mapping'] = {
                    str(category): k for k, category in enumerate(oe.categories_[0])
                }
                column_info[column]['r_encode_mapping'] = {
                    str(k): str(category) for k, category in enumerate(oe.categories_[0])
                }

            elif self.is_discrete_continuous(df[column]) == 'continuous variable':
                for k, v in df[column].describe().items():
                    column_info[column][str(k)] = v
                del column_info[column]['count']

                # 缺失值非常多
                if df[column].isnull().mean() > 0.2:
                    df = df.drop(columns=[column])
                    column_info[column]['nan_processing'] = 'drop'
                    continue

                # 缺失值中等, 不多
                elif 0.2 >= df[column].isnull().mean() > 0:
                    df[column] = df[column].fillna(df[column].mode().iloc[0])
                    column_info[column]['nan_processing'] = 'fill mode'

                else:
                    column_info[column]['nan_processing'] = 'no operation'

                df[column] = StandardScaler().fit_transform(df[[column]])

        preprocessing_config['column_info'] = column_info
        save_dict_2_json(preprocessing_config, self.preprocessing_config_path)
        df.to_csv(self.processed_data_path, index=False)

    def save_correlation_matrix(self):
        df = pd.read_csv(self.processed_data_path)
        plt.figure()
        correlation_matrix = df.corr()
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('Correlation Matrix')
        plt.savefig(self.correlation_matrix_path, bbox_inches='tight', transparent=True)

    def save_2_database(self):
        payload = {
            'file_id': self.file_id,
            'file_name': self.uploaded_file.filename,
            'project_name': self.project_name,
            'mission_type': 'unknown',
            'train_status': 'not start',
            'raw_data_path': self.raw_data_path,
            'processed_data_path': self.processed_data_path,
            'preprocessing_config_path': self.preprocessing_config_path,
            'correlation_matrix_path': self.correlation_matrix_path,
            'cv_res_path': self.cv_res_path,
        }
        DatabaseOperator.insert(CSV, payload)

    @staticmethod
    def is_discrete_continuous(s):
        return 'discrete variable' if 15 > len(s.value_counts()) or s.dtype == 'object' else 'continuous variable'
