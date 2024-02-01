import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class DataframeOperator:
    def __init__(self, file_path, file_name, project_name):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.file_name = file_name
        self.project_name = project_name

    def check_feature(self, features):
        columns = self.df.columns.to_numpy()
        feature_pos_list = []
        for feature in features:
            feature_pos = np.where(columns == feature)[0]
            assert feature_pos.size > 0, "feature not found"
            feature_pos_list.append(str(feature_pos[0]))
        return ','.join(feature_pos_list)

    def check_label(self, label):
        columns = self.df.columns.to_numpy()
        label_pos = np.where(columns == label)[0]
        assert label_pos.size > 0, "label not found"
        return str(label_pos[0])

    def check_mission_type(self, mission_type):
        assert mission_type in ['regression', 'classification'], "mission type error, regression and classification"
        return "1" if mission_type == 'classification' else "0"

    def data_preprocess(self):
        columns = self.df.columns
        for column in columns:
            series = self.df[column]
            if 15 > len(series.value_counts()) or series.dtype == 'object':
                self.df[column] = OrdinalEncoder().fit_transform(self.df[[column]])
            else:
                self.df[column] = StandardScaler().fit_transform(self.df[[column]])

        self.df.to_csv(self.file_path)

    def save_info(self, info_path):
        result = {
            'file_name': self.file_name,
            'project_name': self.project_name,
            'row_num': self.df.shape[0],
            'column_num': self.df.shape[1],
            'columns': list(map(str, self.df.columns)),
            'column_info': {},
        }

        columns = self.df.columns
        for column in columns:
            series = self.df[column]
            if series.dtype == 'object' or 15 > len(series.value_counts()):
                value_counts = series.value_counts()
                result['column_info'][str(column)] = {str(k): v for k, v in value_counts.items()}
                result['column_info'][str(column)]['column_class'] = 'discrete variable'
            else:
                describe = series.describe()
                result['column_info'][str(column)] = {str(k): round(v, 2) for k, v in describe.items()}
                result['column_info'][str(column)]['column_class'] = 'continuous variable'

            result['column_info'][str(column)]['nan'] = int(series.isna().sum())
            result['column_info'][str(column)]['total'] = len(series)

        with open(info_path, 'w') as file:
            json.dump(result, file)

    def save_pic(self, pic_path):
        correlation_matrix = self.df.corr()
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('correlation_matrix')
        plt.savefig(pic_path, bbox_inches='tight', transparent=True)
