import pandas as pd


def load(fname, lines=1000):
    my_data = df = pd.read_csv(fname, )
    return my_data.head(lines)
