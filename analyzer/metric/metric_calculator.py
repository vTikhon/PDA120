import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricCalculator:

    @staticmethod
    def show_metrics(y_test, y_pred):
        # Коэффициент детерминации (R_square)
        print(f'R²: {(r2_score(y_test, y_pred)):.3f}')
        # Средняя абсолютная ошибка (Mean Absolute Error)
        print(f'MAE: {(mean_absolute_error(y_test, y_pred)):.2f}')
        # Средняя абсолютная процентная ошибка (Mean Absolute Percentage Error)
        print(f'MAPE: {(np.mean(np.abs((y_test - y_pred) / y_test)) * 100):.0f}%')
        # Корень из средней квадратичной ошибки (Root Mean Square Error)
        print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.0f}')
        # Среднеквадратичная ошибка (Mean Square Error)
        print(f'MSE: {mean_squared_error(y_test, y_pred):.0f}')
