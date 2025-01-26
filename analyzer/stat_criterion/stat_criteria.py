from scipy.stats import kruskal
from scipy.stats import mannwhitneyu


class StatCriteria:

    def _prepare_groups(self, df, df_column, target):
        if df_column not in df.columns:
            raise KeyError(f"Колонка '{df_column}' отсутствует в DataFrame.")
        groups = []
        for unique_value in df[df_column].unique():
            groups.append(df[df[df_column] == unique_value][target])
        return groups

    # U-критерий Манна-Уитни используется для сравнения различий между двумя независимыми выборками,
    # когда распределение выборки не является нормальным, а размеры выборки малы
    # Нулевая гипотеза (H0): Распределения двух выборок равны (отсутствуют статистически значимые различия).
    def mannwhitneyu(self, df, df_column, target):
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) != 2:
            raise ValueError("Для теста Манна-Уитни требуется ровно две группы.")
        stat, p_value = mannwhitneyu(*groups)
        print('Mann-Whitney U test')
        print('U_statistic =', stat)

        if p_value < 0.05:
            print(f"Распределения двух выборок не равны (p-value = {p_value:.3f}).")
        elif p_value >= 0.05:
            print(f"Распределения двух выборок равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

        return stat, p_value

    # Критерий Краскела - Уоллиса (непараметрический для непрерывных данных)
    # Критерий Краскела - Уоллиса используемый для определения, есть ли статистически значимые различия
    # между медианами трех или более независимых групп
    # Нулевая гипотеза (H0): Медианы всех групп равны (отсутствуют статистически значимые различия).
    def kruskal(self, df, df_column, target):
        if df_column not in df.columns:
            raise KeyError(f"Колонка '{df_column}' отсутствует в DataFrame.")
        groups = self._prepare_groups(df, df_column, target)
        if len(groups) < 2:
            raise ValueError("Для теста Краскела-Уоллиса требуется хотя бы две группы.")
        stat, p_value = kruskal(*groups)
        print('kruskal')
        print('statistic =', stat)

        if p_value < 0.05:
            print(f"Медианы всех групп не равны (p-value = {p_value:.3f}).")
        elif p_value >= 0.05:
            print(f"Медианы всех групп равны (p-value = {p_value:.3f}).")
        else:
            print(f"анализ не проведён (p-value = {p_value:.3f}).")

        return stat, p_value



