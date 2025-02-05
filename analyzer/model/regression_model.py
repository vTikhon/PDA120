import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
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
        return y_train, y_test

    def Lasso(self, X_train, X_test, y_train, y_test,
              alpha=0.01):
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        y_pred = pd.DataFrame(lasso.predict(X_test), columns=y_test.columns)
        print("Коэффициенты Lasso-регрессии:", lasso.coef_)
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        return y_train, y_test

    def Ridge(self, X_train, X_test, y_train, y_test,
              alpha=0.01):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred = pd.DataFrame(ridge.predict(X_test), columns=y_test.columns)
        print("Коэффициенты Ridge-регрессии:", ridge.coef_)
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        return y_train, y_test

    def GLM(self, X_train, X_test, y_train, y_test,
            prepend=False, family=sm.families.Gamma(link=sm.families.links.Log())):
        X_train_plus_const = sm.add_constant(X_train, prepend=prepend)
        X_test_plus_const = sm.add_constant(X_test, prepend=prepend)
        model = sm.GLM(y_train, X_train_plus_const, family=family)
        result = model.fit()
        y_pred = result.predict(X_test_plus_const)
        print(result.summary())
        MetricCalculator.show_regression_metrics(y_test, y_pred)
        return y_train, y_test

