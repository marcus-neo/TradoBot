"""Module containing the generator functions for training and evaulation."""
import os
import shutil
import multiprocessing as mp
import csv
import pandas as pd
import numpy as np
from utils.parameters import Parameters
from utils import utils


def train_generator():
    """Create the training dataset."""
    if os.path.isdir(Parameters.IMAGE_OUTPUT_DIRECTORY):
        shutil.rmtree(Parameters.IMAGE_OUTPUT_DIRECTORY)
    os.mkdir(Parameters.IMAGE_OUTPUT_DIRECTORY)
    with mp.Pool(processes=mp.cpu_count()) as process_pool:
        # Collect Ticker List
        loaded_tickers = np.genfromtxt(
            Parameters.TICKER_LIST_DIRECTORY, delimiter=",", dtype=str
        )
        ticker_list = np.random.choice(
            loaded_tickers, Parameters.SAMPLES, replace=True
        )
        print("Downloading Data...")
        hist_list = process_pool.starmap(
            utils.get_ticker,
            zip(
                ticker_list,
                [Parameters.SAMPLE_WIN_SIZE] * len(ticker_list),
            ),
        )
        print("Filtering Data...")
        hist_list = list(
            filter(
                lambda x: len(x[1].index) > 0.95 * Parameters.SAMPLE_WIN_SIZE,
                hist_list,
            )
        )
        print(f"Total Tickers Obtained: {len(hist_list)}")
        hist_list = [item[1].to_numpy() for item in hist_list]
        sampled_list = process_pool.starmap(
            utils.window_sample,
            zip(
                hist_list,
                [Parameters.LOOKBACK + Parameters.LOOKAHEAD] * len(hist_list),
            ),
        )
        sampled_list = process_pool.map(utils.add_indicators, sampled_list)
        sampled_list = [
            np.split(item, [Parameters.LOOKBACK]) for item in sampled_list
        ]
        split_data = np.array(sampled_list, dtype=object)
        groundtruth_data = split_data[:, 1]
        training_data = np.stack(split_data[:, 0])
        groundtruth_data = np.stack(split_data[:, 1])

        # processing and exporting groundtruths:
        print("Processing and Creating Label Map....")
        groundtruths = process_pool.starmap(
            utils.classify, zip(groundtruth_data, range(len(groundtruth_data)))
        )
        fields = ["filename", "xmin", "ymin", "xmax", "ymax", "label"]
        with open(
            os.path.join(Parameters.LABEL_OUTPUT_DIRECTORY, "labels.csv"),
            "w",
            encoding="ascii",
        ) as outfile:
            write = csv.writer(outfile)
            write.writerow(fields)
            write.writerows(groundtruths)

        # creating training images
        print("Creating Training Images...")
        process_pool.starmap(
            utils.create_training_image,
            zip(training_data, range(len(training_data))),
        )
    print("Completed.")


def test_generator():
    """Create the dataset for evaluations."""
    if os.path.isdir(Parameters.TEST_OUTPUT_DIRECTORY):
        shutil.rmtree(Parameters.TEST_OUTPUT_DIRECTORY)
    os.mkdir(Parameters.TEST_OUTPUT_DIRECTORY)
    ticker_csv = pd.read_csv(Parameters.TICKER_LIST_DIRECTORY, header=None)
    ticker_list = ticker_csv.to_numpy().flatten()

    with mp.Pool(processes=mp.cpu_count()) as process_pool:

        print("Downloading Data...")
        hist_list = process_pool.starmap(
            utils.get_ticker,
            zip(ticker_list, [Parameters.LOOKBACK] * len(ticker_list)),
        )

        print("Filtering Data...")
        hist_list = list(
            filter(
                lambda x: (
                    len(x[1].index) > 0.90 * Parameters.LOOKBACK, hist_list
                )
            )
        )
        print(f"Total Tickers Obtained: {len(hist_list)}")
        names = [item[0] for item in hist_list]
        test_data = [item[1].to_numpy() for item in hist_list]
        test_data = process_pool.map(utils.add_indicators, test_data)

        # Creating Test Images:
        print("Creating Test Images...")
        process_pool.starmap(
            utils.create_test_image,
            zip(test_data, names),
        )

    print("Completed.")


if __name__ == "__main__":
    test_generator()
    # train_generator()
