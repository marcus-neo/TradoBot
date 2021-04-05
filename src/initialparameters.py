import pandas as pd
import yfinance as yf
import numpy as np
import functools as ft

# import keras
import argparse
import random
import datetime
from classify import Classify


def initialparameters(timeSeries, number=5):
    indexList = [i for i in range(100)]
    outputList = [([0] * (number * 5))] * 100

    for j in range(100):
        if j >= number:
            initialise = [0] * (number * 5)
            for i in range(number):
                initialise[i * number] = timeSeries.iloc[j - i - 1]["Open"]
                initialise[i * number + 1] = timeSeries.iloc[j - i - 1]["High"]
                initialise[i * number + 2] = timeSeries.iloc[j - i - 1]["Low"]
                initialise[i * number + 3] = timeSeries.iloc[j - i - 1]["Close"]
                initialise[i * number + 4] = timeSeries.iloc[j - i - 1]["Volume"]
            outputList[j] = initialise
    return outputList


if __name__ == "__main__":
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1Y")
    input = hist.head(100)
    print(initialparameters(input))
