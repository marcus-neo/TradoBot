import pandas as pd
import yfinance as yf
import numpy as np

# import keras
import argparse
import random
import datetime
from classify import Classify


def chooseSample(tickerList):
    sample = tickerList.sample().values.flatten()[0]
    ticker = yf.Ticker(sample)
    hist = ticker.history(period="max")
    length = len(hist.index)
    if length < 100:
        chooseSample(tickerList)
    else:
        choices = length - 100
        randomIndex = random.randint(0, choices)
        return sample, hist.iloc[randomIndex : randomIndex + 100]


# def args_parser():
#     parser = argparse.ArgumentParser(
#         description="Datature Open Source Prediction Script"
#     )
#     parser.add_argument(
#         "--inputModel", help="Path to folder that contains the model", required=True
#     )
#     parser.add_argument(
#         "--config", help="Tester Configurations", default="0"
#     )
#     return parser.parse_args()


def backtester():
    # Load argument variables
    # args = args_parser()
    tickerLst = pd.read_csv("../StockTickers/TickerNames.csv", header=None)
    print(
        "Loading random ticker for test. Ignore any subsequent errors until test is loaded."
    )
    while True:
        try:
            [sampleTicker, sample] = chooseSample(tickerLst)
            break
        except:
            pass
    listOfDateTime = list(sample.index)
    listOfDates = [d.strftime("%m-%d-%Y") for d in listOfDateTime]
    print("Ticker Loaded.")
    print(
        "The test sample is "
        + sampleTicker
        + " from "
        + str(listOfDates[0])
        + " to "
        + str(listOfDates[-1])
        + "."
    )
    # print(sample)
    output = Classify(sample["Close"].to_numpy())
    print(output)
    # model = keras.models.load_model(args.inputModel)


if __name__ == "__main__":
    backtester()
