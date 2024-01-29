import copy
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


class TrainModelBase:
    CLASSIFICATION_SCORE_NAME = ['accuracy', 'average_precision', 'recall_weighted']
    REGRESSION_SCORE_NAME = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

    def __init__(self, csv_path: str, label_column: int, feature_columns: list, class_or_reg: int):
        self.csv_path = csv_path
        self.file_id = csv_path.replace('.csv', '').replace('\\', '/').split('/')[-1]
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.class_or_reg = class_or_reg

        self.model_name = None
        self.model = None
        self.score_names = self.CLASSIFICATION_SCORE_NAME if class_or_reg else self.REGRESSION_SCORE_NAME
        self.scores = None

        self.dataframe = None
        self.label = None
        self.features = None

    def read_csv(self):
        self.dataframe = pd.read_csv(self.csv_path)
        self.label = self.dataframe.iloc[:, self.label_column]
        self.features = self.dataframe.iloc[:, self.feature_columns]

    def cross_validation(self):
        res = cross_validate(self.model, self.features, self.label, scoring=self.score_names)
        if self.class_or_reg:
            self.scores = {
                "accuracy": res['test_accuracy'].mean(),
                "precision": res['test_average_precision'].mean(),
                "recall": res['test_recall_weighted'].mean()
            }
        else:
            self.scores = {
                "mae": -res['test_neg_mean_absolute_error'].mean(),
                "mse": -res['test_neg_mean_squared_error'].mean(),
                "r_square": res['test_r2'].mean()
            }

        self.scores = {k: round(float(v), 2) for k, v in self.scores.items()}

    def store_result(self):
        conn = sqlite3.connect('instance/ml_database.db')
        cursor = conn.cursor()

        if self.class_or_reg:
            insert_data_sql = f'INSERT INTO `classification` (`file_id`, `model`, `accuracy`, `precision`, `recall`) VALUES (?, ?, ?, ?, ?)'
            data_to_insert = (
                self.file_id, self.model_name, self.scores['accuracy'], self.scores['precision'], self.scores['recall'])
        else:
            insert_data_sql = f'INSERT INTO `regression` (`file_id`, `model`, `mae`, `mse`, `r_square`) VALUES (?, ?, ?, ?, ?)'
            data_to_insert = (
                self.file_id, self.model_name, self.scores['mae'], self.scores['mse'], self.scores['r_square'])

        cursor.execute(insert_data_sql, data_to_insert)
        conn.commit()
        cursor.close()
        conn.close()

    def plot_result(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.label, test_size=0.33, random_state=42)
        model = copy.deepcopy(self.model)
        model.fit(X_train, y_train)

        if self.class_or_reg:
            cm = confusion_matrix(y_test, model.predict(X_test))
            plot = sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
            plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')

        else:
            df = pd.DataFrame({'y_true': y_test, 'y_predict': model.predict(X_test)})
            df = df.sort_values(by=['y_true'], ignore_index=True)
            sns.lineplot(df)

        png_path = self.csv_path.split('/')
        png_path[-1] = self.model_name + '.png'

        plt.title('confusion_matrix')
        plt.savefig('/'.join(png_path), bbox_inches='tight', transparent=True)
