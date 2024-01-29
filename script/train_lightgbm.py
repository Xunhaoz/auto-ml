import sys
from train_model_base import TrainModelBase
from lightgbm import LGBMClassifier, LGBMRegressor


class LightGBM(TrainModelBase):

    def __init__(self, csv_path: str, label_column: int, feature_columns: list, class_or_reg: int):
        super().__init__(csv_path, label_column, feature_columns, class_or_reg)
        self.model_name = 'lightgbm'
        self.model = LGBMClassifier(verbose=-1) if class_or_reg else LGBMRegressor(verbose=-1)


if __name__ == '__main__':
    csv_path = sys.argv[1]
    label_column = int(sys.argv[2])
    feature_columns = list(map(int, sys.argv[3].split(',')))
    class_or_reg = int(sys.argv[4])

    lightgbm = LightGBM(csv_path, label_column, feature_columns, class_or_reg)
    lightgbm.read_csv()
    lightgbm.cross_validation()
    lightgbm.store_result()
    lightgbm.plot_result()

