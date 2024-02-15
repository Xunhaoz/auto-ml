import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from package.database_operator import *
from script.json_toolkit import *

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, VotingClassifier
from joblib import dump, load

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


class AiOperator:
    def __init__(self, file_id, label_column, feature_columns, mission_type, app):
        self.file_id = file_id
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.mission_type = mission_type
        self.app = app

        self.csv = DatabaseOperator.select_one(CSV, {'file_id': self.file_id}, app=app)
        self.df = pd.read_csv(self.csv.processed_data_path)

    def cross_validation(self):
        models, scores = self.get_model_score()
        X, y = self.df[self.feature_columns], self.df[self.label_column]
        result = {
            'label': self.label_column,
            'feature': self.feature_columns,
            'models': [model.__class__.__name__ for model in models],
            'results': {}
        }
        for model in models:
            model_res = cross_validate(model, X, y, scoring=scores)
            result['results'][model.__class__.__name__] = self.standardize_score(model_res)
        save_dict_2_json(result, self.csv.cv_res_path)

    def plot_validation(self):
        models, scores = self.get_model_score()
        X_train, X_test, y_train, y_test = train_test_split(
            self.df[self.feature_columns], self.df[self.label_column], test_size=0.33, random_state=42
        )
        for model in models:
            plt.figure()
            model.fit(X_train, y_train)
            if self.mission_type == 'classification':
                cm = confusion_matrix(y_test, model.predict(X_test))
                plot = sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
                plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
                plt.title('Confusion Matrix')
            else:
                df = pd.DataFrame({'y_true': y_test, 'y_predict': model.predict(X_test)})
                df = df.sort_values(by=['y_true'], ignore_index=True)
                sns.lineplot(df)
                plt.title('Prediction')

            png_path = self.csv.raw_data_path.split('/')
            png_path[-1] = model.__class__.__name__ + '.png'
            plt.savefig('/'.join(png_path), transparent=True, format="png")

    def train_model(self):
        models, scores = self.get_model_score()
        X, y = self.df[self.feature_columns], self.df[self.label_column]
        for model in models:
            model_path = self.csv.raw_data_path.split('/')
            model_path[-1] = model.__class__.__name__ + '.joblib'
            model.fit(X, y)
            dump(model, '/'.join(model_path))

    def fool_poof(self):
        assert self.mission_type in ['classification', 'regression'], 'mission type error'
        assert len(set(self.feature_columns)) == len(self.feature_columns), 'should not have duplicate columns'

        columns = self.df.columns
        for column in self.feature_columns + [self.label_column]:
            assert column in columns, 'either label or feature columns is error'

    def get_model_score(self):
        if self.mission_type == 'classification':
            models = [XGBClassifier(), LGBMClassifier(verbose=-1), CatBoostClassifier(verbose=False)]
            scores = ['accuracy', 'average_precision', 'recall_weighted']

        else:
            models = [XGBRegressor(), LGBMRegressor(verbose=-1), CatBoostRegressor(verbose=False)]
            scores = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

        return models, scores

    @staticmethod
    def standardize_score(score):
        res = {}
        for k, v in score.items():
            if 'neg' in k:
                res[k.replace('neg_', '')] = -v.mean()
            else:
                res[k] = v.mean()
        return res
