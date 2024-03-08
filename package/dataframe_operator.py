from pathlib import Path

from script.json_toolkit import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class DataFrameOperator:
    def __init__(self, file_dir, preprocessed_config):
        self.file_dir = Path(file_dir)
        self.preprocessed_config = json.loads(preprocessed_config)

    def preprocessing_train_csv(self):
        df = pd.read_csv(self.file_dir / Path('raw_data.csv'))
        for column, column_info in self.preprocessed_config.items():
            if column_info['column_class'] == 'drop variable':
                df = df.drop(columns=[column])
                continue

            if self.preprocessed_config[column]['na_processing'] == 'fill mode':
                df[column] = df[column].fillna(df[column].mode()[0])

            elif self.preprocessed_config[column]['na_processing'] == 'fill unknown':
                df[column] = df[column].fillna('Unknown')

            if self.preprocessed_config[column]['column_class'] == 'continuous variable':
                df[column] = (df[column] - df[column].mean()) / df[column].std()
            elif self.preprocessed_config[column]['column_class'] == 'discrete variable':
                df[column] = LabelEncoder().fit_transform(df[[column]])
        df.to_csv(self.file_dir / Path('preprocessed_data.csv'), index=False)
        return df

    def preprocessing_predict_csv(self, feature=None):
        train_df = pd.read_csv(self.file_dir / Path('raw_data.csv'))
        predict_df = pd.read_csv(self.file_dir / Path('predict_raw_data.csv'))

        train_df = train_df[feature]
        predict_df = predict_df[feature]

        for column, column_info in self.preprocessed_config.items():
            if column not in feature:
                continue

            if column_info['column_class'] == 'drop variable':
                predict_df = predict_df.drop(columns=[column])
                continue

            if self.preprocessed_config[column]['na_processing'] == 'fill mode':
                predict_df[column] = predict_df[column].fillna(train_df[column].mode()[0])

            elif self.preprocessed_config[column]['na_processing'] == 'fill unknown':
                predict_df[column] = predict_df[column].fillna('Unknown')

            if self.preprocessed_config[column]['column_class'] == 'continuous variable':
                predict_df[column] = (predict_df[column] - train_df[column].mean()) / train_df[column].std()
            elif self.preprocessed_config[column]['column_class'] == 'discrete variable':
                le = LabelEncoder()
                le.fit(train_df[[column]])
                predict_df[column] = le.transform(predict_df[[column]])
        predict_df.to_csv(self.file_dir / Path('predict_preprocessed_data.csv'), index=False)
        return predict_df

    def reverse_preprocessing_predict_csv(self, result_path, feature):
        train_df = pd.read_csv(self.file_dir / Path('raw_data.csv'))
        result_df = pd.read_csv(result_path)

        for column, column_info in self.preprocessed_config.items():
            if column not in feature:
                continue

            if self.preprocessed_config[column]['column_class'] == 'continuous variable':
                result_df[column] = result_df[column] * train_df[column].std() + train_df[column].mean()

            elif self.preprocessed_config[column]['column_class'] == 'discrete variable':
                le = LabelEncoder()
                le.fit(train_df[[column]])
                result_df[column] = le.inverse_transform(result_df[[column]])

        result_df.to_csv(result_path, index=False)
        return result_df

    def save_correlation_matrix(self):
        df = pd.read_csv(self.file_dir / Path('preprocessed_data.csv'))
        plt.figure()
        correlation_matrix = df.corr()
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('Correlation Matrix')
        plt.savefig(self.file_dir / Path('corr_matrix.png'), bbox_inches='tight', transparent=True)

        return (self.file_dir / Path('corr_matrix.png')).__str__()
