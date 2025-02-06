import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from analyzer.metric import MetricCalculator


class RegressionModel:

    def __init__(self):
        pass


    def PolynomialFeatures(self, X_train, X_test, y_train, y_test,
                           degree=2, include_bias=False):
        # Трансформируем Х в полиномиальные признаки
        poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_train = pd.DataFrame(poly_features.fit_transform(X_train),
                               columns=poly_features.get_feature_names_out(X_train.columns))
        X_test = pd.DataFrame(poly_features.transform(X_test),
                              columns=poly_features.get_feature_names_out(X_test.columns))
        return X_train, X_test, y_train, y_test

    def OLS(self, X_train, X_test, y_train, y_test,
            prepend=False):
        X_train_plus_const = sm.add_constant(X_train, prepend=prepend)
        X_test_plus_const = sm.add_constant(X_test, prepend=prepend)
        model = OLS(y_train, X_train_plus_const).fit()
        y_pred = pd.DataFrame(model.predict(X_test_plus_const), columns=y_test.columns)
        print(model.summary())
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        return y_test, y_pred, model

    def Lasso(self, X_train, X_test, y_train, y_test,
              alpha=0.01):
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        print('__________')
        print("Коэффициенты Lasso-регрессии:")
        print(*[f"{feature}: {coef:.2f}" for feature, coef in zip(X_train.columns, model.coef_.flatten())], sep='\n')
        return y_test, y_pred, model

    def Ridge(self, X_train, X_test, y_train, y_test,
              alpha=0.01):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        print('__________')
        print("Коэффициенты Ridge-регрессии:")
        print(*[f"{feature}: {coef:.2f}" for feature, coef in zip(X_train.columns, model.coef_.flatten())], sep='\n')
        return y_test, y_pred, model

    def GLM(self, X_train, X_test, y_train, y_test,
            prepend=False, family=sm.families.Gamma(link=sm.families.links.Log())):
        X_train_plus_const = sm.add_constant(X_train, prepend=prepend)
        X_test_plus_const = sm.add_constant(X_test, prepend=prepend)
        model = sm.GLM(y_train, X_train_plus_const, family=family)
        result = model.fit()
        y_pred = result.predict(X_test_plus_const)
        print(result.summary())
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        return y_test, y_pred, model

    def SVR(self, X_train, X_test, y_train, y_test,
            kernel='rbf', C=1.0, epsilon=0.1):
        model = svm.SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        return y_test, y_pred, model
