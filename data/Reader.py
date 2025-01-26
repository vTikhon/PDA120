import pandas as pd


class Reader:

    @staticmethod
    def reader(link):
        df = pd.read_csv(link)
        df.columns = map(lambda x: x.replace(" ", "_").lower(), df.columns)
        return df
