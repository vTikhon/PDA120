import pandas as pd


class Reader:

    @staticmethod
    def read_csv(link):
        # Чтение CSV
        df = pd.read_csv(link)
        df = df.sample(len(df)).reset_index(drop=True)
        df.columns = (df.columns
                      .str.replace(r"[ \-.,]", "_", regex=True)
                      .str.lower())
        return df

    @staticmethod
    def replace_symbol_in_data(df, symbol):
        return df.replace(symbol, 'other')