import pandas as pd
from analyzer.metric import MetricCalculator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


class ClassificationModel:

    def __init__(self):
        pass


    def LogisticRegression(self, X_train, X_test, y_train, y_test,
                           penalty='l1', solver='liblinear', max_iter=1000, C=0.1):
        model = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter, C=C)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_classification_metrics(y_test, y_pred)
        print('__________')
        print("Коэффициенты регуляризации:")
        print(*[f"{feature}: {coef:.2f}" for feature, coef in zip(X_train.columns, model.coef_.flatten())], sep='\n')
        return y_test, y_pred, model

    def GaussianNB(self, X_train, X_test, y_train, y_test):
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_classification_metrics(y_test, y_pred)
        return y_test, y_pred, model

    def KNeighborsClassifier(self, X_train, X_test, y_train, y_test,
                             n_neighbors=5, metric='euclidean'):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_classification_metrics(y_test, y_pred)
        return y_test, y_pred, model

    def SVC(self, X_train, X_test, y_train, y_test,
            kernel='rbf'):
        model = svm.SVC(kernel=kernel)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        MetricCalculator.show_classification_metrics(y_test, y_pred)
        return y_test, y_pred, model

    def DecisionTreeClassifier(self, X_train, X_test, y_train, y_test,
                               max_depth=2, criterion='gini'):
        model = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        tree.plot_tree(model)
        print(classification_report(y_test, y_pred))
        MetricCalculator.show_classification_metrics(y_test, y_pred)
        return y_test, y_pred, model