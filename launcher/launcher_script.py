import pandas as pd
from sklearn.metrics import r2_score

from analyzer.metric import MetricCalculator
from analyzer.model import RegressionModel
from data import DataPreparation
from data.io import Reader
from data.normalizer import Normalizer


class LaunchPredict:

    def launch_predict(self, data):
        X_new = DataPreparation.create_dataframe(data)
        csv_path = Reader.get_csv_path('notebooks\dataset', 'RK_554B_data.csv')
        df_original = Reader.read_csv(csv_path)
        df = df_original.copy().reset_index(drop=True)

        # подготавливаем данные
        target = ['compensation_resistor']
        exclude_features = ['compensation_resistor', 'temperature_of_peregib' , 'frequency_stability']
        X_train, X_test, y_train, y_test = DataPreparation().train_test_split(df, exclude_features, target)

        # проводим нормализацию
        X_train, X_test, y_train, y_test = Normalizer().min_max_scaling(X_train, X_test, y_train, y_test)

        # Применим метод опорных векторов
        y_test, y_pred, model = RegressionModel().SVR(X_train, X_test, y_train, y_test, kernel='rbf', C=1.0, epsilon=0.1)

        # Производим расчёт y_pred от поступившего X_new
        R_square_metric = r2_score(y_test, y_pred)
        if R_square_metric >= 0.95:
            y_pred = pd.DataFrame(model.predict(X_new), columns=y_test.columns)
            return y_pred.iloc[0, 0]

        return f"Model should be recharged, because R_square = {R_square_metric:.3f} < 0.95"



