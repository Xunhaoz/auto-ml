from pathlib import Path
import zipfile

import seaborn as sns
import matplotlib.pyplot as plt

from package.database_operator import *
from script.json_toolkit import *

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from joblib import dump, load

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


class AiOperator:
    def __init__(self, file_id, dataframe, label_column, feature_columns, mission_type, app):
        self.file_id = file_id
        self.dataframe = dataframe
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.mission_type = mission_type
        self.app = app

    def cross_validation(self):
        models, scores = self.get_model_score()
        X, y = self.dataframe[self.feature_columns], self.dataframe[self.label_column]

        result = {
            'label': self.label_column,
            'feature': self.feature_columns,
            'mission_type': self.mission_type,
            'models': [model.__class__.__name__ for model in models],
            'results': {},
        }

        for model in models:
            model_res = cross_validate(model, X, y, scoring=scores)
            result['results'][model.__class__.__name__] = self.standardize_score(model_res)

        DatabaseOperator.update(
            CSV, {'file_id': self.file_id}, {'train_result': json.dumps(result)}, self.app
        )

    def plot_validation(self):
        models, scores = self.get_model_score()
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataframe[self.feature_columns], self.dataframe[self.label_column], test_size=0.33, random_state=42
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

            plt.savefig(Path(f'file/{self.file_id}/{model.__class__.__name__}.png'), transparent=True, format="png")

    def train_model(self):
        models, scores = self.get_model_score()
        X, y = self.dataframe[self.feature_columns], self.dataframe[self.label_column]
        for model in models:
            model.fit(X, y)
            dump(model, Path(f'file/{self.file_id}/{model.__class__.__name__}.joblib'))

    def get_predict(self):
        models, scores = self.get_model_score()
        X = self.dataframe[self.feature_columns]
        prediction_files = []
        dir_path = Path(f'file/{self.file_id}')
        for model in models:
            model_path = dir_path / Path(f'{model.__class__.__name__}.joblib')
            prediction_path = dir_path / Path(f'{model.__class__.__name__}_prediction.csv')
            model = load(model_path)
            res = model.predict(X)
            self.dataframe[self.label_column] = res
            self.dataframe.to_csv(prediction_path, index=False)
            prediction_files.extend([model_path, prediction_path])


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
                res[k.replace('neg_', '')] = round(float(-v.mean()), 4)
            else:
                res[k] = round(float(v.mean()), 4)
        return res
