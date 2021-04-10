import pandas as pd
import yfinance as yf
import numpy as np

# import keras
import argparse
import random
import datetime
from classify import Classify
from normalisePandas import normPd
from initialparameters import initialparameters


def getCloseSeries(ticker, number=5):
    data = yf.Ticker(ticker)
    hist = data.history(period="max")
    hist = hist.drop(columns=["Dividends", "Stock Splits"])
    length = len(hist.index)
    if length < number:
        raise ValueError
    else:
        return normPd(hist)


WINDOW_SIZE = 5
tickerLst = pd.read_csv("../StockTickers/TickerNames2.csv", header=None)
convertedTickerLst = tickerLst.values.flatten()
df = pd.DataFrame()
i = 0
for ticker in convertedTickerLst:
    if i > 10:
        break
    else:
        print("Executing " + str(i) + "out of " + str(len(convertedTickerLst)))
        # try:
        closeSeries = getCloseSeries(ticker, number=WINDOW_SIZE)
        output = Classify(closeSeries, WINDOW_SIZE)
        input = initialparameters(closeSeries, WINDOW_SIZE)
        input = input[WINDOW_SIZE:]
        input = input[: len(output)]
        for j in range(len(output)):
            input[j].append(output[j])
        df = pd.concat([df, pd.DataFrame(input)], ignore_index=True)
        print(df)
        # except:
        #     continue
    i += 1
print(df)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("train.csv")
