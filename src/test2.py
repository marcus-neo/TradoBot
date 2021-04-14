import numpy as np
import tulipy as ti
import yfinance as yf
import pandas as pd
import sys
import backtester

tickerLst = pd.read_csv("../StockTickers/TickerNames2.csv", header=None)
[sampleTicker, sample] = backtester.chooseSample(tickerLst)

print(sampleTicker)
print(sample)

open_list = sample["Open"].to_numpy()
high_list = sample["High"].to_numpy()
low_list = sample["Low"].to_numpy()
close_list = sample["Close"].to_numpy()
volume_list = sample["Volume"].to_numpy()
class_method = getattr(ti, "ad")
print(class_method.__name__)


def test(**kwargs):
    for key, value in kwargs.items():
        param = ""
        if "open" in value[0]:
            param += "open=open_list"
        if "high" in value[0]:
            if param:
                param += ", "
            param += "high=high_list"
        if "low" in value[0]:
            if param:
                param += ", "
            param += "low=low_list"
        if "close" in value[0]:
            if param:
                param += ", "
            param += "close=close_list"
        if "volume" in value[0]:
            if param:
                param += ", "
            param += "volume=volume_list"
        print(param)
        ad = eval("class_method"+"("+param+")")
        print(ad, len(ad))


# param = "high=high, low=low, close=close, volume=volume"
test(ti=(["high", "low", "close", "volume"], 1))
# class_method(high, low, close, volume)
# ad = eval("class_method"+"("+param+")")
# print(ad, len(ad))
