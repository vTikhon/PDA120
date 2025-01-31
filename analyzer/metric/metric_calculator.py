import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MetricCalculator:

    @staticmethod
    def show_regression_metrics(y_test, y_pred):
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.flatten()
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values.flatten()
        # Коэффициент детерминации (R_square)
        print(f'R²: {r2_score(y_test, y_pred):.3f}')
        # Средняя абсолютная ошибка (Mean Absolute Error)
        print(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
        # Средняя абсолютная процентная ошибка (Mean Absolute Percentage Error)
        print(f'MAPE: {(np.mean(np.abs((y_test - y_pred) / y_test)) * 100):.0f}%')
        # Корень из средней квадратичной ошибки (Root Mean Square Error)
        print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.0f}')
        # Среднеквадратичная ошибка (Mean Square Error)
        print(f'MSE: {mean_squared_error(y_test, y_pred):.0f}')

    @staticmethod
    def show_classification_metrics(y_test, y_pred, y_prob=None):
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.flatten()
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values.flatten()
        # Метрика корректности прогнозов — доля верно предсказанных классов
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        # Метрика точности — доля объектов, предсказанных как положительные
        print(f'Precision: {precision_score(y_test, y_pred, average="weighted"):.2f}')
        # Метрика полноты — доля правильно предсказанных положительных объектов от всех фактических положительных объектов
        print(f'Recall: {recall_score(y_test, y_pred, average="weighted"):.2f}')
        # F1-score — гармоническое среднее между Precision и Recall
        print(f'F1-score: {f1_score(y_test, y_pred, average="weighted"):.2f}')
        # AUC-ROC — оценка качества вероятностных предсказаний
        if y_prob is not None:
            print(f'ROC-AUC: {roc_auc_score(y_test, y_prob, multi_class="ovr"):.2f}')
