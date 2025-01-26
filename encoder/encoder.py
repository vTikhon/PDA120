from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


class Encoder:
    def __init__(self):
        pass

    def bolean_one_column_encoder(self, df_feature, false_feature_parameter):
        df_feature = df_feature.apply(lambda x: 1 if x == false_feature_parameter else 0)
        return df_feature

    def leaveOneOutEncoder(self, df_feature, df_target):
        encoder = ce.LeaveOneOutEncoder()
        df_feature = encoder.fit_transform(df_feature, df_target)
        return df_feature

    def labelEncoder(self, df_feature):
        # Закодируем категориальную переменную ремонт c помощью Label Encoder
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_feature)
        df_feature = label_encoder.transform(df_feature)
        return df_feature
