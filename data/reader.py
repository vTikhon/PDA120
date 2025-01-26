import pandas as pd


class Reader:

    @staticmethod
    def read_csv(link):
        # Чтение CSV
        df = pd.read_csv(link)
        df.columns = map(lambda x: x.replace(" ", "_").lower(), df.columns)
        return df
