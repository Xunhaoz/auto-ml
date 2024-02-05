import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from package.database_models import *

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, VotingClassifier
from package.database_models import *

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

    def cross_validation(self):
        models, scores = self.get_model_score()
        self.update_sql('pending', 'train_status')
        self.update_sql(self.mission_type, 'mission_type')
        try:
            df = pd.read_csv(self.select_sql().processed_file_path)
            X, y = df[self.feature_columns], df[self.label_column]
            result = {}
            for model in models:
                model_res = cross_validate(model, X, y, scoring=scores)
                result[model.__class__.__name__] = self.standardize_score(model_res)
            self.save_json(result)
        except Exception as e:
            self.update_sql('failed', 'train_status')

    def plot_validation(self):
        models, scores = self.get_model_score()
        try:
            df = pd.read_csv(self.select_sql().processed_file_path)
            X_train, X_test, y_train, y_test = train_test_split(
                df[self.feature_columns], df[self.label_column], test_size=0.33, random_state=42
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

                png_path = self.select_sql().processed_file_path.split('/')
                png_path[-1] = model.__class__.__name__ + '.png'
                plt.savefig('/'.join(png_path), transparent=True, format="png")
            self.update_sql('finished', 'train_status')
        except Exception as e:
            self.update_sql('failed', 'train_status')

    def predict(self):
        ...

    def fool_poof(self):
        assert self.mission_type in ['classification', 'regression'], 'mission type error'

        assert len(set(self.feature_columns)) == len(self.feature_columns), 'should not have duplicate columns'

        columns = pd.read_csv(self.select_sql().processed_file_path).columns
        for column in self.feature_columns + [self.label_column]:
            assert column in columns, 'either label or feature columns is error'

    def get_model_score(self, prediction=False):
        if self.mission_type == 'classification':
            models = [XGBClassifier(), LGBMClassifier(verbose=-1), CatBoostClassifier(verbose=False)]
            scores = ['accuracy', 'average_precision', 'recall_weighted']
            if prediction:
                return VotingClassifier(estimators=[(model.__calss__.__name__, model) for model in models])

        else:
            models = [XGBRegressor(), LGBMRegressor(verbose=-1), CatBoostRegressor(verbose=False)]
            scores = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
            if prediction:
                return VotingRegressor(estimators=[(model.__calss__.__name__, model) for model in models])

        return models, scores

    def save_json(self, msg):
        with open(self.select_sql().file_cv_res_path, 'w') as f:
            json.dump(msg, f)

    def update_sql(self, msg, col):
        with self.app.app_context():
            csv = CSV.query.filter_by(file_id=self.file_id).first()
            if col == 'predict_status':
                csv.predict_status = msg
            elif col == 'train_status':
                csv.train_status = msg
            elif col == 'mission_type':
                csv.mission_type = msg
            db.session.commit()

    def select_sql(self):
        with self.app.app_context():
            csv = CSV.query.filter_by(file_id=self.file_id).first()
        return csv

    @staticmethod
    def standardize_score(score):
        res = {}
        for k, v in score.items():
            if 'neg' in k:
                res[k.replace('neg_', '')] = -round(float(v.mean()), 2)
            else:
                res[k] = round(float(v.mean()), 2)
        return res


def training_pipeline(*args):
    ai_op = AiOperator(*args)
    ai_op.fool_poof()
    ai_op.cross_validation()
    ai_op.plot_validation()
