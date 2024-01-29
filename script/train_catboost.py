import sys
from train_model_base import TrainModelBase
from catboost import CatBoostRegressor, CatBoostClassifier


class CatBoost(TrainModelBase):

    def __init__(self, csv_path: str, label_column: int, feature_columns: list, class_or_reg: int):
        super().__init__(csv_path, label_column, feature_columns, class_or_reg)
        self.model_name = 'catboost'
        self.model = CatBoostClassifier(verbose=False) if class_or_reg else CatBoostRegressor(verbose=False)


if __name__ == '__main__':
    csv_path = sys.argv[1]
    label_column = int(sys.argv[2])
    feature_columns = list(map(int, sys.argv[3].split(',')))
    class_or_reg = int(sys.argv[4])

    catboost = CatBoost(csv_path, label_column, feature_columns, class_or_reg)
    catboost.read_csv()
    catboost.cross_validation()
    catboost.store_result()
    catboost.plot_result()
