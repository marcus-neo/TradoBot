"""Module containing all utility functions."""
import os
import random
from typing import List
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
from utils.indicator_params import IndicatorDict

indicators = IndicatorDict().indicator_dict


def add_indicators(hist: np.ndarray) -> np.ndarray:
    """Add all indicator outputs from indicator dictionary into the data array.

    :param hist:    The current raw historical data array.
    :return:        The updated array with all indicator outputs from the
                    indicator dictionary added to it.
    """
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


def get_ticker(ticker_name: str, window: int) -> pd.DataFrame:
    """Retrieve the historical data from Yahoo Finance.

    :param ticker_name: The name of the ticker to be retrieved.
    :param window:      The total window size to be retrieved.
    :return:            The historical data represented by pandas dataframe.
    """
    return (
        ticker_name,
        yf.Ticker(ticker_name)
        .history(period=f"{window}d")
        .drop(columns=["Dividends", "Stock Splits"], errors="ignore"),
    )


def ticker_sampler(ticker_list_directory: str) -> str:
    """Sample a single ticker from the list of tickers.

    :param ticker_list_directory:   Path of the file containing all tickers.
    :return:                        A name of the sampled ticker.
    """
    ticker_list = pd.read_csv(ticker_list_directory)
    return ticker_list.sample().values.flatten()[0]


def window_sample(hist: np.ndarray, window: int) -> np.ndarray:
    """Sample a random window from the given historical data.

    :param hist:    The historical data.
    :param window:  The window size to sample.
    :return:        The sampled historical data.
    """
    random_range = len(hist) - window
    sampled_index = random.randint(0, random_range - 1)
    return hist[sampled_index: sampled_index + window]


def classify(
    data: np.ndarray, index: int
) -> List[str, int, int, int, int, str]:
    """Classify whether the given historical data is buy, sell or no action.

    :param data:    The historical data.
    :param index:   The name of the file that contains the data
                    (We assume its the image index).
    :return:        A list containing the image name that corresponds to the
                    data, and the action.
    """
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
    if value is None:
        value = "no_action"
    return [
        f"{index}.jpg",
        0,
        0,
        Parameters.IMAGE_WIDTH,
        Parameters.IMAGE_HEIGHT,
        value,
    ]


def resize(filepath: str) -> None:
    """Open and update an image by resizing it according to the preset dims.

    :param filename:    The path of the image to be resized.
    """
    image = Image.open(filepath)
    image = image.resize((Parameters.IMAGE_WIDTH, Parameters.IMAGE_HEIGHT))
    image.save(filepath)


def create_training_image(data: np.ndarray, index: int) -> None:
    """Create a heatmap from the training data, and save it.

    :param data:    The historical data, potentially with indicator outputs.
    :param index:   The data's index number, used for saving.
    """
    # This minmaxscalar normalizes each column:
    #   minimum of the column -> 0
    #   maximum of the column -> 1
    #   all other values in between becomes normalized accordingly.
    min_max_scaler = preprocessing.MinMaxScaler()
    fig1, figaxis = plt.subplots(
        figsize=(11, 11),
        frameon=False,
    )
    figaxis.set_axis_off()
    image_name = os.path.join(
        Parameters.IMAGE_OUTPUT_DIRECTORY, f"{index}.jpg"
    )

    # General pipeline function:
    #   output of a function goes directly into the input of the next function,
    #   and this continues to the last function, where its output is returned.
    pipe(
        data,
        min_max_scaler.fit_transform,
        lambda x: sb.heatmap(x, cbar=False),
        lambda _: plt.savefig(image_name, bbox_inches="tight", pad_inches=0),
    )
    resize(image_name)
    plt.close(fig1)


def create_test_image(data: np.ndarray, index: int) -> None:
    """Create a heatmap from the test data, and save it.

    :param data:    The historical data, potentially with indicator outputs.
    :param index:   The data's index number, used for saving.
    """
    # This minmaxscalar normalizes each column:
    #   minimum of the column -> 0
    #   maximum of the column -> 1
    #   all other values in between becomes normalized accordingly.
    min_max_scaler = preprocessing.MinMaxScaler()
    fig1, figaxis = plt.subplots(
        figsize=(11, 11),
        frameon=False,
    )
    figaxis.set_axis_off()
    image_name = os.path.join(Parameters.TEST_OUTPUT_DIRECTORY, f"{index}.jpg")

    # General pipeline function:
    #   output of a function goes directly into the input of the next function,
    #   and this continues to the last function, where its output is returned.
    pipe(
        data,
        min_max_scaler.fit_transform,
        lambda x: sb.heatmap(x, cbar=False),
        lambda _: plt.savefig(image_name, bbox_inches="tight", pad_inches=0),
    )
    resize(image_name)
    plt.close(fig1)
