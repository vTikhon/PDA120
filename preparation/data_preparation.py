from sklearn.model_selection import train_test_split


class DataPreparation:

    def __init__(self):
        pass

    @staticmethod
    def prepare_data(df, exclude_features, target):
        # Разделяем данные на X, y
        X = df.drop(exclude_features, axis=1)
        y = df[target]
        # Разделим данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test