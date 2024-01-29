import sys
from train_model_base import TrainModelBase
from xgboost import XGBClassifier, XGBRegressor


class XgBoost(TrainModelBase):

    def __init__(self, csv_path: str, label_column: int, feature_columns: list, class_or_reg: int):
        super().__init__(csv_path, label_column, feature_columns, class_or_reg)
        self.model_name = 'xgboost'
        self.model = XGBClassifier() if class_or_reg else XGBRegressor()


if __name__ == '__main__':
    csv_path = sys.argv[1]
    label_column = int(sys.argv[2])
    feature_columns = list(map(int, sys.argv[3].split(',')))
    class_or_reg = int(sys.argv[4])

    xgboost = XgBoost(csv_path, label_column, feature_columns, class_or_reg)
    xgboost.read_csv()
    xgboost.cross_validation()
    xgboost.store_result()
    xgboost.plot_result()

