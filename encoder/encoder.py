from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import pandas as pd


class Encoder:
    def __init__(self):
        pass

    def bolean_one_column_encoder(self, df_feature, false_feature_parameter):
        df_feature_encoded = df_feature.apply(lambda x: 1 if x == false_feature_parameter else 0)
        return df_feature_encoded

    def labelEncoder(self, df_feature):
        label_encoder = LabelEncoder()
        df_feature_encoded = label_encoder.fit_transform(df_feature)
        return df_feature_encoded

    def oneHotEncoder(self, df_feature):
        # Применяем One-Hot Encoding для категориального признака
        one_hot_encoder = OneHotEncoder(sparse=False)  # sparse=False для получения DataFrame, а не матрицы
        df_feature_encoded = one_hot_encoder.fit_transform(df_feature.values.reshape(-1, 1))
        # Возвращаем результат как DataFrame с подходящими колонками
        df_encoded = pd.DataFrame(df_feature_encoded, columns=one_hot_encoder.get_feature_names_out())
        return df_encoded

    def binaryEncoder(self, df_feature):
        encoder = ce.BinaryEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    def helmertEncoder(self, df_feature):
        encoder = ce.HelmertEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    def backwardDifferenceEncoder(self, df_feature):
        encoder = ce.BackwardDifferenceEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    def hashingEncoder(self, df_feature, n_components=8):
        encoder = ce.HashingEncoder(n_components=n_components)
        df_feature_encoded = encoder.fit_transform(df_feature)
        return df_feature_encoded

    def targetEncoder(self, df_feature, df_target):
        encoder = ce.TargetEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature, df_target)
        return df_feature_encoded

    def leaveOneOutEncoder(self, df_feature, df_target):
        encoder = ce.LeaveOneOutEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature, df_target)
        return df_feature_encoded

    def jamesSteinEncoder(self, df_feature, df_target):
        encoder = ce.JamesSteinEncoder()
        df_feature_encoded = encoder.fit_transform(df_feature, df_target)
        return df_feature_encoded

