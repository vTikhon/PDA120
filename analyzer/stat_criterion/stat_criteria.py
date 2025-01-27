from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, chisquare, norm, ttest_1samp, ttest_ind
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
import pandas as pd
import numpy as np


class StatCriteria:

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def _prepare_groups(self, df, df_column, target):
        if df_column is None or target is None:
            raise ValueError("Оба параметра df_column и target должны быть указаны.")
        groups = df_column.dropna().unique()
        if len(groups) < 2:
            raise ValueError("Для анализа требуется минимум 2 группы.")
        if df_column not in df.columns:
            raise KeyError(f"Колонка '{df_column}' отсутствует в DataFrame.")
        groups = []
        for unique_value in df[df_column].unique():
            groups.append(df[df[df_column] == unique_value][target])
        return groups

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=1
    # Одновыборочный z-критерий (параметрический для непрерывных данных)
    # z-критерий используется для проверки того,
    # является ли среднее значение генеральной совокупности (N=1, то есть одной выборки)
    # меньше, больше или равно некоторому определенному значению.
    # Нулевая гипотеза (H0): Среднее значение выборки равно заданному значению.
    def z_criteria(self, df_column, hypothesized_mean, std):
        # Рассчитаем выборочное среднее и размер выборки
        sample_mean = np.mean(df_column)
        sample_size = len(df_column)
        # Рассчитаем стандартную ошибку
        standard_error = std / np.sqrt(sample_size)
        # Z-статистика
        z_stat = (sample_mean - hypothesized_mean) / standard_error
        # p-value для двухстороннего теста
        p_value = 2 * norm.sf(abs(z_stat))
        print(f"Test z-statistic")
        print(f'z-statistic = {z_stat:.3f}')
        if p_value < self.alpha:
            print(f"Среднее значение выборки не равно заданному значению (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Среднее значение выборки равно заданному значению (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return z_stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=1
    # Критерий Стьюдента (одновыброчный)
    # Нулевая гипотеза (H0): Среднее значение выборки равно заданному значению.
    def ttest_1samp(self, df_column, hypothesized_mean):
        df_column = df_column.dropna()
        if len(df_column) == 0:
            raise ValueError("Выборка пуста после исключения пропусков.")
        t_stat, p_value = ttest_1samp(df_column, hypothesized_mean)
        print(f"One Samples T-test")
        print(f't-statistic = {t_stat:.3f}')
        if p_value < self.alpha:
            print(f"Среднее значение выборки не равно заданному значению (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Среднее значение выборки равно заданному значению (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return t_stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=2
    # Критерий Стьюдента (параметрический для непрерывных данных)
    # это статистический метод, который позволяет сравнивать средние значения двух выборок и на основе результатов теста делать
    # заключение о том, различаются ли они друг от друга статистически или нет.
    # Нулевая гипотеза (H0): Средние значения в двух выборках равны (отсутствуют статистически значимые различия).
    def ttest_ind(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) != 2:
            raise ValueError("Для критерия Стьюдента требуется ровно две группы.")
        t_stat, p_value = ttest_ind(*groups)
        print("Independent Samples T-test")
        print(f't-statistic = {t_stat:.3f}')
        if p_value < self.alpha:
            print(f"Средние значения в двух выборках не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Средние значения в двух выборках равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return t_stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N=2
    # U-критерий Манна-Уитни используется для сравнения различий между ДВУМЯ (N=2) независимыми выборками,
    # когда распределение выборки не является нормальным, а размеры выборки малы
    # Нулевая гипотеза (H0): Распределения двух выборок равны (отсутствуют статистически значимые различия).
    def mannwhitneyu(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) != 2:
            raise ValueError("Для теста Манна-Уитни требуется ровно две группы.")
        stat, p_value = mannwhitneyu(*groups)
        print('Mann-Whitney U test')
        print(f'U_statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Распределения двух выборок не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Распределения двух выборок равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ДАННЫЕ) N>=3
    # Критерий Краскела - Уоллиса (непараметрический для непрерывных данных)
    # Критерий Краскела - Уоллиса используемый для определения, есть ли статистически значимые различия
    # между медианами ТРЁХ И БОЛЕЕ (n>=3) независимых групп
    # Нулевая гипотеза (H0): Медианы всех групп равны (отсутствуют статистически значимые различия).
    def kruskal(self, df, df_column, target):
        if df_column not in df.columns:
            raise KeyError(f"Колонка '{df_column}' отсутствует в DataFrame.")
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) < 2:
            raise ValueError("Для теста Краскела-Уоллиса требуется хотя бы две группы.")
        stat, p_value = kruskal(*groups)
        print('kruskal')
        print(f'statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Медианы всех групп не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Медианы всех групп равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=1
    # Критерий хи-квадрат
    # Хи-квадрат критерия согласия — используется для определения того,
    # следует ли категориальная переменная (N=1 - то есть одна выборка) гипотетическому распределению.
    # Нулевая гипотеза (H0): Наблюдаемые частоты согласуются с ожидаемыми частотами (нет статистически значимых различий).
    def chisquare(self, observed, expected):
        if len(observed) != len(expected):
            raise ValueError("Массивы наблюдаемых и ожидаемых частот должны быть одинаковой длины.")
        chisq_stat, p_value = chisquare(observed, expected)
        print("Chi-square test statistic")
        print(f'chisq_statistic = {chisq_stat:.3f}')
        if p_value < self.alpha:
            print(f"Наблюдаемые частоты не согласуются с ожидаемыми частотами (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Наблюдаемые частоты согласуются с ожидаемыми частотами (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return chisq_stat, p_value

    # НЕЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=2
    # Критерий хи-квадрат
    # Критерий независимости хи-квадрат — используется для определения наличия значимой связи
    # между ДВУМЯ (N=2) категориальными переменными
    # Нулевая гипотеза (H0): Две переменные независимы (отсутствуют статистически значимые связи).
    def chi2_contingency(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) != 2:
            raise ValueError("Для критерия независимости хи-квадрат требуется ровно две группы.")
        contingency_table = pd.crosstab(*groups)
        _, p_value, _, _ = chi2_contingency(contingency_table)
        print('Chi-square test')
        if p_value < self.alpha:
            print(f"Две переменные зависимы (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Две переменные независимы (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return p_value

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ИЛИ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=2
    # Критерий Вилкоксона (непараметрический для непрерывных и порядковых данных)
    # Критерий Вилкоксона используется для сравнения ДВУХ (N=2) связанных (зависимых) выборок
    # по количественному или порядковому признаку.
    # Нулевая гипотеза (H0): Различия между парами значений равны (отсутствуют статистически значимые различия).
    def wilcoxon(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) != 2:
            raise ValueError("Для теста Вилкоксона требуется ровно две группы.")
        if len(df_column) != len(target):
            raise ValueError("Выборки должны быть одинаковой длины для теста Уилкоксона.")
        stat, p_value = wilcoxon(*groups)
        print('Wilcoxon test')
        print(f'Wilcoxon-statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Различия между парами значений не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Различия между парами значений равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ НЕПРЕРЫВНЫЕ ИЛИ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N>=3
    # Критерий Фридмана (непараметрический для непрерывных и порядковых данных)
    # Критерий Фридмана используется для сравнения ТРЁХ И БОЛЕЕ (N>=3) связанных (зависимых) выборок
    # по количественному или порядковому признаку.
    # Нулевая гипотеза (H0): Распределения во всех группах равны (отсутствуют статистически значимые различия).
    def friedmanchisquare(self, df, df_column, target):
        if df_column not in df.columns:
            raise KeyError(f"Колонка '{df_column}' отсутствует в DataFrame.")
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) < 2:
            raise ValueError("Для теста Фридмана требуется хотя бы две группы.")
        stat, p_value = friedmanchisquare(*groups)
        print('Friedman test')
        print('statistic = {stat:.3f}')
        if p_value < self.alpha:
            print(f"Распределения во всех группах не равны (p-value = {p_value:.3f}).")
        elif p_value >= self.alpha:
            print(f"Распределения во всех группах равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")
        return stat, p_value

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N=2
    # Критерий Мак-Нимара (для категорильных зависимых данных)
    # Тест Макнемара используется для определения наличия статистически значимой разницы в пропорциях
    # между ПАРНЫМИ (N=2) данными.
    # Нулевая гипотеза (H0): Доли согласия в двух связанных группах равны (отсутствуют статистически значимые различия).
    def mcnemar(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) != 2:
            raise ValueError("Для теста Мак-Нимара требуется ровно две группы.")
        # Создание контингентной таблицы
        cross_tab = pd.crosstab(*groups)
        print("McNemar's test")
        result = mcnemar(cross_tab, exact=True)  # Используется exact=True для точного теста (если требуется)
        print(result)

    # ЗАВИСИМЫЕ ОТ ВРЕМЕНИ (ПРИ ЭТОМ КАТЕГОРИАЛЬНЫЕ ДАННЫЕ) N>=3
    # Q-тест Кокрана (для категорильных зависимых данных)
    # Q-тест Кокрана (Cochran's Q test) — это статистический тест,
    # используемый для определения наличия различий в доли успеха в нескольких (N>=3) связанных группах.
    # Q-тест Кокрана применяется, когда данные являются бинарными (успех/неудача) и измерены повторно в различных условиях.
    def cochrans_q(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) < 2:
            raise ValueError("Для Q-теста Кокрана требуется хотя бы две группы.")
        cross_tab = pd.crosstab(*groups)
        print("Cochran's Q test: ")
        result = cochrans_q(cross_tab)
        print(result)



