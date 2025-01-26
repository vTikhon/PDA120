from data import Reader
import pandas as pd
pd.set_option('display.max_columns', None)  # Показать все столбцы
pd.set_option('display.width', 1000)        # Увеличить ширину отображения

if __name__ == '__main__':
    print('\nчитаем датасет в память из внешнего источника:')
    df = Reader.reader('https://raw.githubusercontent.com/Semendyeav/datasets/main/PDA120_Moscow_Price.csv')
    print(df.head(10))

    print('\nпосмотрим на пропуски в данных:')
    print(df.isna().sum())

