import os
import uuid
import json
from package.database_models import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class CsvOperator:
    def __init__(self, uploaded_file, project_name):
        self.uploaded_file, self.project_name = uploaded_file, project_name
        self.file_id = str(uuid.uuid4())
        self.file_dir = f'file/{self.file_id}/'
        self.ori_file_path = f'file/{self.file_id}/{self.uploaded_file.filename}'
        self.processed_file_path = f'file/{self.file_id}/{self.file_id}.csv'
        self.info_path = f'file/{self.file_id}/{self.file_id}.json'
        self.pic_path = f'file/{self.file_id}/{self.file_id}.png'
        self.file_cv_res_path = f'file/{self.file_id}/cv_{self.file_id}.json'

    def save_csv(self):
        os.makedirs(self.file_dir)
        self.uploaded_file.save(self.ori_file_path)
        self.uploaded_file.save(self.processed_file_path)

    def preprocessed(self):
        df = pd.read_csv(self.ori_file_path)

        for column in df.columns:
            # 離散變量
            if self.is_discrete_continuous(df[column]) == 'discrete variable':
                # 缺失值非常多
                if df[column].isnull().mean() > 0.2:
                    df = df.drop(columns=[column])
                    continue

                # 缺失值中等
                if 0.2 >= df[column].isnull().mean() > 0.1:
                    df[column] = df[column].fillna('Unknown')
                # 缺失值不多
                else:
                    df[column] = df[column].fillna(df[column].mode().iloc[0])

                df[column] = OrdinalEncoder().fit_transform(df[[column]])

            # 連續變量
            else:
                # 缺失值非常多
                if df[column].isnull().mean() > 0.2:
                    df = df.drop(columns=[column])
                    continue

                # 缺失值不多
                df[column] = df[column].fillna(df[column].mode().iloc[0])
                df[column] = StandardScaler().fit_transform(df[[column]])

        df.to_csv(self.processed_file_path, index=False)

    def save_info(self):
        df = pd.read_csv(self.ori_file_path)

        result = {
            'file_name': self.uploaded_file.filename,
            'project_name': self.project_name,
            'row_num': df.shape[0],
            'column_num': df.shape[1],
            'columns': list(map(str, df.columns)),
            'column_info': {},
        }

        for column in df.columns:

            # 離散變量
            if self.is_discrete_continuous(df[column]) == 'discrete variable':
                value_counts = df[column].value_counts()
                result['column_info'][str(column)] = {str(k): v for k, v in value_counts.items()}
                result['column_info'][str(column)]['column_class'] = 'discrete variable'

            else:
                describe = df[column].describe()
                result['column_info'][str(column)] = {str(k): round(v, 2) for k, v in describe.items()}
                result['column_info'][str(column)]['column_class'] = 'continuous variable'

            result['column_info'][str(column)]['nan'] = int(df[column].isnull().sum())
            result['column_info'][str(column)]['total'] = len(df[column])

        with open(self.info_path, 'w') as file:
            json.dump(result, file)

    def save_pic(self):
        df = pd.read_csv(self.processed_file_path)
        plt.figure()
        correlation_matrix = df.corr()
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('Correlation Matrix')
        plt.savefig(self.pic_path, bbox_inches='tight', transparent=True)

    def insert_db(self):
        db.session.add(CSV(
            file_id=self.file_id,
            file_name=self.uploaded_file.filename,
            project_name=self.project_name,
            mission_type='unknown',
            train_status='not start',
            predict_status='not start',
            ori_file_path=self.ori_file_path,
            processed_file_path=self.processed_file_path,
            file_info_path=self.info_path,
            file_pic_path=self.pic_path,
            file_cv_res_path=self.file_cv_res_path
        ))
        db.session.commit()

    @staticmethod
    def is_discrete_continuous(s):
        return 'discrete variable' if 15 > len(s.value_counts()) or s.dtype == 'object' else 'continuous variable'


def upload_csv_pipeline(uploaded_file, project_name):
    csv_op = CsvOperator(uploaded_file, project_name)
    csv_op.save_csv()
    csv_op.preprocessed()
    csv_op.save_info()
    csv_op.save_pic()
    csv_op.insert_db()
    return csv_op.file_id
