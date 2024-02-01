import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from package.database_models import *

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


def np2float(arr):
    return round(float(arr.mean()), 2)


def training_fool_poof(csv_path, label_column, feature_columns, class_or_reg):
    assert class_or_reg in ['classification', 'regression'], 'mission type error'

    df = pd.read_csv(csv_path).dropna()
    for column in feature_columns + [label_column]:
        assert column in df.columns, 'either label or feature columns is error'

    label_column_series = df[label_column]
    if label_column_series.dtype == "object" or 15 > len(label_column_series.value_counts()):
        assert class_or_reg == 'classification', 'mission type error'
    else:
        assert class_or_reg == 'regression', 'mission type error'

    if class_or_reg == 'classification':
        models = [XGBClassifier(), LGBMClassifier(verbose=-1), CatBoostClassifier(verbose=False)]
        scores = ['accuracy', 'average_precision', 'recall_weighted']
    else:
        models = [XGBRegressor(), LGBMRegressor(verbose=-1), CatBoostRegressor(verbose=False)]
        scores = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

    return df, models, scores


def training_plotting_pipline(*args):
    file_id, csv_path, label_column, feature_columns, class_or_reg, app = args
    df, models, scores = training_fool_poof(csv_path, label_column, feature_columns, class_or_reg)

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns], df[label_column], test_size=0.33, random_state=42)

    for model in models:
        plt.figure()

        model.fit(X_train, y_train)
        if class_or_reg == 'classification':
            cm = confusion_matrix(y_test, model.predict(X_test))
            plot = sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
            plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
            plt.title('confusion_matrix')
        else:
            df = pd.DataFrame({'y_true': y_test, 'y_predict': model.predict(X_test)})
            df = df.sort_values(by=['y_true'], ignore_index=True)
            sns.lineplot(df)
            plt.title('prediction')

        png_path = csv_path.split('/')
        png_path[-1] = model.__class__.__name__ + '.png'

        plt.savefig('/'.join(png_path), transparent=True, format="png")


def training_pipline(*args):
    file_id, csv_path, label_column, feature_columns, class_or_reg, app = args
    df, models, scores = training_fool_poof(csv_path, label_column, feature_columns, class_or_reg)

    X, y = df[feature_columns], df[label_column]

    for model in models:
        res = cross_validate(model, X, y, scoring=scores)

        if class_or_reg == 'classification':
            table = Classification(
                file_id=file_id,
                model=model.__class__.__name__,
                accuracy=np2float(res['test_accuracy']),
                precision=np2float(res['test_accuracy']),
                recall=np2float(res['test_accuracy']),
            )
        else:
            table = Regression(
                file_id=file_id,
                model=model.__class__.__name__,
                mse=np2float(-res['test_neg_mean_absolute_error']),
                mae=np2float(-res['test_neg_mean_squared_error']),
                r_square=np2float(res['test_r2']),
            )

        with app.app_context():
            db.session.add(table)
            csv = CSV.query.filter_by(file_id=file_id).first()
            csv.status = 'finish'
            db.session.commit()
