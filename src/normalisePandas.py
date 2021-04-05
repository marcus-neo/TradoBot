import pandas as pd
from sklearn import preprocessing

"""Takes in Pandas Dataframe and normalises every column"""


def normPd(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    return df
