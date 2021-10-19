import os
import random
import yfinance as yf
import pandas as pd
import numpy as np
from PIL import Image
import tulipy as ti
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from toolz.functoolz import pipe
from utils.parameters import Parameters
from utils.indicator_params import indicator_dict

indicators = indicator_dict().indicator_dict


def add_indicators(hist):
    hist_length = len(hist)
    all_columns = {
        "open": hist[:, 0],
        "high": hist[:, 1],
        "low": hist[:, 2],
        "close": hist[:, 3],
        "volume": hist[:, 4],
    }

    for indicator in indicators:
        indicator_name = indicator["name"]
        required_columns = indicator["primary_columns"]
        period_list = (
            indicator["period_list"] if "period_list" in indicator else None
        )
        func = getattr(ti, indicator_name)
        params = [
            np.ascontiguousarray(all_columns[col_name])
            for col_name in required_columns
        ]
        if period_list is not None:
            for period in period_list:
                params.append(period)
        output = func(*params)
        output = list(output) if isinstance(output, tuple) else [output]
        for item in output:
            hist = np.c_[
                hist,
                np.pad(
                    item,
                    (hist_length - len(item), 0),
                    "constant",
                    constant_values=0,
                ),
            ]
    return hist


def get_ticker(ticker_name, window):
    return (
        ticker_name,
        yf.Ticker(ticker_name)
        .history(period=f"{window}d")
        .drop(columns=["Dividends", "Stock Splits"], errors="ignore"),
    )


def ticker_sampler(ticker_list_directory):
    ticker_list = pd.read_csv(ticker_list_directory)
    return ticker_list.sample().values.flatten()[0]


def window_sample(hist, window):
    random_range = len(hist) - window
    sampled_index = random.randint(0, random_range - 1)
    return hist[sampled_index : sampled_index + window]


def classify(data, index):
    high = data[1:, 1]
    low = data[1:, 2]
    keypoint = data[0, 3]
    comparison_points = data[1:]
    value = None
    for comparison_point in comparison_points:
        high = comparison_point[1]
        low = comparison_point[2]
        hit_tp = high / keypoint * 100 - 100 > Parameters.SUCCESSFUL_TRADE_PERC
        hit_sl = 100 - low / keypoint * 100 > Parameters.SUCCESSFUL_TRADE_PERC
        if hit_tp and hit_sl:
            value = "unclear"
            break
        if hit_tp:
            value = "long"
            break
        if hit_sl:
            value = "short"
            break
    if value == None:
        value = "no_action"
    return [
        f"{index}.jpg",
        0,
        0,
        Parameters.IMAGE_WIDTH,
        Parameters.IMAGE_HEIGHT,
        value,
    ]


def resize(filename):
    image = Image.open(filename)
    image = image.resize((Parameters.IMAGE_WIDTH, Parameters.IMAGE_HEIGHT))
    image.save(filename)


def create_training_image(item, index):
    min_max_scaler = preprocessing.MinMaxScaler()
    fig1, ax = plt.subplots(
        figsize=(11, 11),
        frameon=False,
    )
    ax.set_axis_off()
    image_name = os.path.join(
        Parameters.IMAGE_OUTPUT_DIRECTORY, f"{index}.jpg"
    )
    pipe(
        item,
        min_max_scaler.fit_transform,
        lambda x: sb.heatmap(x, cbar=False),
        lambda _: plt.savefig(image_name, bbox_inches="tight", pad_inches=0),
    )
    resize(image_name)
    plt.close(fig1)


def create_test_image(item, index):
    min_max_scaler = preprocessing.MinMaxScaler()
    fig1, ax = plt.subplots(
        figsize=(11, 11),
        frameon=False,
    )
    ax.set_axis_off()
    image_name = os.path.join(Parameters.TEST_OUTPUT_DIRECTORY, f"{index}.jpg")
    pipe(
        item,
        min_max_scaler.fit_transform,
        lambda x: sb.heatmap(x, cbar=False),
        lambda _: plt.savefig(image_name, bbox_inches="tight", pad_inches=0),
    )
    resize(image_name)
    plt.close(fig1)
