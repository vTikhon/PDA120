import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MetricCalculator:

    @staticmethod
    def show_metrics(y_test, y_pred):
        print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
        print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')