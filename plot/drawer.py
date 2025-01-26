import seaborn as sns
import matplotlib.pyplot as plt


class Drawer:

    @staticmethod
    def draw_kde_hist_boxplot(df):
        for column in df.select_dtypes(include='number').columns:
            plt.figure(figsize=(16, 4))
            sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})

            plt.subplot(1, 3, 1)
            sns.kdeplot(data=df, x=column, color='green')

            plt.subplot(1, 3, 2)
            sns.histplot(data=df, x=column, color='blue')

            plt.subplot(1, 3, 3)
            sns.boxplot(data=df, x=column, color='red')

            plt.show()

