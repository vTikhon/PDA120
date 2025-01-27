import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer


class Normalizer:
    def __init__(self):
        pass

    @staticmethod
    def log_norm(X_train, X_test, y_train, y_test):
        if np.any(X_train < 0) or np.any(X_test < 0) or np.any(y_train < 0) or np.any(y_test < 0):
            raise ValueError("Все значения в данных должны быть неотрицательными для использования log1p.")
        X_train_normalized = np.log1p(X_train)
        y_train_normalized = np.log1p(y_train)
        X_test_normalized = np.log1p(X_test)
        y_test_normalized = np.log1p(y_test)
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

    @staticmethod
    def min_max_scaling(X_train, X_test, y_train, y_test, feature_range=(0, 1)):
        # Нормализация признаков
        scaler_X = MinMaxScaler(feature_range=feature_range)
        X_train_normalized = scaler_X.fit_transform(X_train)
        X_test_normalized = scaler_X.transform(X_test)  # Нормализуем тест по параметрам train
        # Нормализация таргета
        scaler_y = MinMaxScaler(feature_range=feature_range)
        y_train_normalized = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test_normalized = scaler_y.transform(y_test.reshape(-1, 1))
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

    @staticmethod
    def standard_scaler(X_train, X_test, y_train, y_test):
        # Нормализация признаков
        scaler_X = StandardScaler()
        X_train_normalized = scaler_X.fit_transform(X_train)
        X_test_normalized = scaler_X.transform(X_test)
        # Нормализация таргета
        scaler_y = StandardScaler()
        y_train_normalized = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test_normalized = scaler_y.transform(y_test.reshape(-1, 1))
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

    @staticmethod
    def robust_scaler(X_train, X_test, y_train, y_test, quantile_range=(25.0, 75.0)):
        # Нормализация признаков
        scaler_X = RobustScaler(quantile_range=quantile_range)
        X_train_normalized = scaler_X.fit_transform(X_train)
        X_test_normalized = scaler_X.transform(X_test)  # Нормализация теста по параметрам train
        # Нормализация таргета
        scaler_y = RobustScaler(quantile_range=quantile_range)
        y_train_normalized = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test_normalized = scaler_y.transform(y_test.reshape(-1, 1))
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

    @staticmethod
    def power_transform(X_train, X_test, y_train, y_test, method='yeo-johnson'):
        # Нормализация признаков
        transformer_X = PowerTransformer(method=method)
        X_train_normalized = transformer_X.fit_transform(X_train)
        X_test_normalized = transformer_X.transform(X_test)
        # Нормализация таргета
        transformer_y = PowerTransformer(method=method)
        y_train_normalized = transformer_y.fit_transform(y_train.reshape(-1, 1))
        y_test_normalized = transformer_y.transform(y_test.reshape(-1, 1))
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

    @staticmethod
    def quantile_transform(X_train, X_test, y_train, y_test, output_distribution='uniform'):
        # Нормализация признаков
        transformer_X = QuantileTransformer(output_distribution=output_distribution)
        X_train_normalized = transformer_X.fit_transform(X_train)
        X_test_normalized = transformer_X.transform(X_test)  # Нормализация теста по параметрам train
        # Нормализация таргета
        transformer_y = QuantileTransformer(output_distribution=output_distribution)
        y_train_normalized = transformer_y.fit_transform(y_train.reshape(-1, 1))
        y_test_normalized = transformer_y.transform(y_test.reshape(-1, 1))
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized