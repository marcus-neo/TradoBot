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


def chooseSample(tickerList):
    sample = tickerList.sample().values.flatten()[0]
    ticker = yf.Ticker(sample)
    hist = ticker.history(period="max")
    hist = hist.drop(columns=["Dividends", "Stock Splits"])
    length = len(hist.index)
    if length < 100:
        chooseSample(tickerList)
    else:
        choices = length - 100
        randomIndex = random.randint(0, choices)
        return sample, normPd(hist.iloc[randomIndex: randomIndex + 100])


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
    tickerLst = pd.read_csv("../StockTickers/TickerNames2.csv", header=None)
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
    # print(sample)
    output = Classify(sample, 5)
    print(output)
    print("Output length = " + str(len(output)))
    input = initialparameters(sample, 5)
    input = input[5:]
    input = input[0: len(output)]
    print(input)
    print("Input length = " + str(len(input)))

    # model = keras.models.load_model(args.inputModel)


if __name__ == "__main__":
    backtester()
