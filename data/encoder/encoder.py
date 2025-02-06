from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import pandas as pd


class Encoder:
    def __init__(self):
        pass

    @staticmethod
    def booleanOneColumnEncoder(df_feature, false_feature_parameter):
        df_feature_encoded = df_feature.apply(lambda x: 1 if x == false_feature_parameter else 0)
        return df_feature_encoded

    @staticmethod
    def labelEncoder(df_feature):
        label_encoder = LabelEncoder()
        df_feature_encoded = label_encoder.fit_transform(df_feature)
        return df_feature_encoded

    @staticmethod
    def oneHotEncoder(df_feature):
        # Применяем One-Hot Encoding для категориального признака
        one_hot_encoder = OneHotEncoder(sparse=False)  # sparse=False для получения DataFrame, а не матрицы
        df_feature_encoded = one_hot_encoder.fit_transform(df_feature.values.reshape(-1, 1))
        # Возвращаем результат как DataFrame с подходящими колонками
        df_encoded = pd.DataFrame(df_feature_encoded, columns=one_hot_encoder.get_feature_names_out())
        return df_encoded

    @staticmethod
    def binaryEncoder(df_feature):
        encoder = ce.BinaryEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    @staticmethod
    def helmertEncoder(df_feature):
        encoder = ce.HelmertEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    @staticmethod
    def backwardDifferenceEncoder(df_feature):
        encoder = ce.BackwardDifferenceEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    @staticmethod
    def hashingEncoder(df_feature, n_components=8):
        encoder = ce.HashingEncoder(n_components=n_components)
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    @staticmethod
    def targetEncoder(df_feature, df_target):
        encoder = ce.TargetEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature, df_target)
        return df_feature_encoded, encoder

    @staticmethod
    def leaveOneOutEncoder(df_feature, df_target):
        encoder = ce.LeaveOneOutEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature, df_target)
        return df_feature_encoded, encoder

    @staticmethod
    def jamesSteinEncoder(df_feature, df_target):
        encoder = ce.JamesSteinEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature, df_target)
        return df_feature_encoded, encoder

