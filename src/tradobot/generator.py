"""Contain the code to generate the training and test sets."""
import random
from datetime import datetime, date, timedelta
import time
import multiprocessing as mp
import pandas as pd
import yfinance as yf

from tradobot.utils.generator_utils import (
    normalize_dataframe,
    custom_arguments,
    classify,
    initial_parameters,
    max_indicator_length
)


class GenerateTest:

    """Create an array of lists.

        The list size is determined by:
            total indicator outputs (rsi, etc) +
            total raw outputs (open, high, low, close, vol)
        The array size is determined by:
            total_window -
            windows discarded by past_window_size considerations -
            windows discarded by indicators -
            windows discarded by zero-tail outputs
            (if last few rows are zeros, they'll be discarded until
            a non-zero output is found. This is to increase the number
            of zero outputs as they are far too common)
    Inputs:
        *args : Additional commands to remove primary rows from output.
            Possible commands are
                "no_open", "no_close", "no_high", "no_low", "no_volume"

        ordered_or_shuffled : Whether the data is ordered or randomly shuffled
        past_window_size : how far in the past that you would want the model
            to consider, in order to make its prediction.

        prediction_length : if the model gives a signal,
            how long in the future does that signal stand?
            (example if model says buy and TP not hit after
            (prediction_length) intervals, then it's signal is wrong)

        random_dates_total_window : The window that will be used for
            each sample, ending with the last available trading date.

        fixed_dates_start : The start date

        fixed_dates_end : The end date

        successful_trade_percent : take-profit percentage
            (for both long and short trades).

        total_samples : Number of random samples in the
            ticker list to sample from.

        ticker_list_directory : file location of the list of tickers.

        **kwargs : Additional commands to add indicators.
            All indicators can be found at
            https://tulipindicators.org/list.
            Currently only support indicator type.

        Add kwargs in the form of:
        YOUR_INDICATOR_NAME={
            # primary columns required to compute the indicator
            "primary_columns": ["high", "low", "close"],
            # number of outputs that the indicator will provide
            "output_columns": 2,
            # add any additional period params in order here
            "period_list": [2, 3, 5],
        },

    Outputs:
        The array of lists to be used for training.
    """

    def __init__(
        self,
        *args,
        past_window_size=5,
        prediction_length=5,
        successful_trade_percent=5.0,
        total_samples=200,
        ticker_name="AAPL",
        total_window=100,
        fixed_dates_start=None,
        fixed_dates_end=None,
        **kwargs
    ):
        """Initialize the Generator class."""
        # Error Catching
        if past_window_size < 5:
            raise ValueError(
                "past_window_size is currently < 5, but is required to be >= 5"
            )
        if prediction_length < 1:
            raise ValueError("prediction_length cannot be 0 or negative.")
        if successful_trade_percent <= 0.0:
            raise ValueError(
                "successful_trade_percent cannot be 0 or negative"
            )
        if total_samples <= 0:
            raise ValueError("total_samples cannot be 0 or negative")
        self.past_window_size = past_window_size
        self.prediction_length = prediction_length
        self.fixed_dates_start = fixed_dates_start
        self.fixed_dates_end = fixed_dates_end
        self.successful_trade_percent = 0.01 * successful_trade_percent
        self.ticker_name = ticker_name
        self.total_samples = total_samples
        self.total_window = total_window
        self.args = args
        self.kwargs = kwargs.items()
        if fixed_dates_end is None and fixed_dates_start is None:
            raise ValueError(
                "either random_dates_total_window or"
                "fix_dates input has to have a value"
            )
        if fixed_dates_end is None:
            raise ValueError("fixed_dates_end needs to be filled in")
        if fixed_dates_start is None:
            raise ValueError("fixed_dates_start needs to be filled in")

        end_datetime = datetime.strptime(fixed_dates_end, "%Y-%m-%d")
        start_datetime = datetime.strptime(fixed_dates_start, "%Y-%m-%d")
        if (
            int((end_datetime - start_datetime).days / 7 * 5)
            < max_indicator_length(self.kwargs)
            + past_window_size
            + prediction_length
            + 1
        ):
            raise ValueError(
                "too small time difference between startdate and enddate."
            )

    def _choose_sample(self):
        start_date = date.today() - timedelta(
            days=int(self.total_window / 5 * 7)
        )
        hist = yf.download(
            self.ticker_name, start=start_date, end=date.today()
        )
        if len(hist.index) == 0:
            raise RuntimeError("Failed Download, empty dataset")
        if hist.isnull().values.any():
            raise RuntimeError("Failed Download, NaN values identified.")
        hist = hist.drop(columns=["Adj Close"])
        return (self.ticker_name, hist)

    def _create_df(self, preprocessed_data, dataframe):
        processed_data = custom_arguments(
            preprocessed_data, self.args, self.kwargs
        )
        normalized_data = normalize_dataframe(processed_data)
        total_columns = len(normalized_data.columns)
        input_data = initial_parameters(
            normalized_data, total_columns, self.past_window_size
        )
        input_data = input_data[self.past_window_size :]  # noqa: E203
        print(input_data[-1])
        dataframe = pd.concat([dataframe, pd.DataFrame(input_data)]).tail(1)
        dataframe = dataframe.sort_index()
        return dataframe

    def generate(self):
        """Generate the test dataset."""
        dataframe = pd.DataFrame()
        ticker, preprocessed_data = self._choose_sample()
        try:
            dataframe = self._create_df(preprocessed_data, dataframe)
        except ValueError as e:
            print(str(e))
            print("ticker: ", ticker)
        return dataframe.fillna(0)


class GenerateTrain:

    """Create an array of lists.

        The list size is determined by:
            total indicator outputs (rsi, etc) +
            total raw outputs (open, high, low, close, vol)
        The array size is determined by:
            total_window -
            windows discarded by past_window_size considerations -
            windows discarded by indicators -
            windows discarded by zero-tail outputs
            (if last few rows are zeros, they'll be discarded until
            a non-zero output is found. This is to increase the number
            of zero outputs as they are far too common)
    Inputs:
        *args : Additional commands to remove primary rows from output.
            Possible commands are
                "no_open", "no_close", "no_high", "no_low", "no_volume"

        ordered_or_shuffled : Whether the data is ordered or randomly shuffled
        past_window_size : how far in the past that you would want the model
            to consider, in order to make its prediction.

        prediction_length : if the model gives a signal,
            how long in the future does that signal stand?
            (example if model says buy and TP not hit after
            (prediction_length) intervals, then it's signal is wrong)

        random_dates_total_window : The window that will be used for
            each sample, ending with the last available trading date.

        fixed_dates_start : The start date

        fixed_dates_end : The end date

        successful_trade_percent : take-profit percentage
            (for both long and short trades).

        total_samples : Number of random samples in the
            ticker list to sample from.

        ticker_list_directory : file location of the list of tickers.

        **kwargs : Additional commands to add indicators.
            All indicators can be found at
            https://tulipindicators.org/list.
            Currently only support indicator type.

        Add kwargs in the form of:
        YOUR_INDICATOR_NAME={
            # primary columns required to compute the indicator
            "primary_columns": ["high", "low", "close"],
            # number of outputs that the indicator will provide
            "output_columns": 2,
            # add any additional period params in order here
            "period_list": [2, 3, 5],
        },

    Outputs:
        The array of lists to be used for training.
    """

    def __init__(
        self,
        *args,
        ordered_or_shuffled="ordered",
        past_window_size=5,
        prediction_length=5,
        successful_trade_percent=5.0,
        total_samples=200,
        ticker_list_directory="../StockTickers/TickerNames.csv",
        random_dates_total_window=None,
        fixed_dates_start=None,
        fixed_dates_end=None,
        **kwargs
    ):
        """Initialize the Generator class."""
        # Error Catching
        if ordered_or_shuffled not in ["shuffled", "ordered"]:
            raise ValueError(
                "invalid entry 'ordered_or_shuffled' parameter."
                "Only 'ordered' or 'shuffled' are accepted."
            )
        if past_window_size < 5:
            raise ValueError(
                "past_window_size is currently < 5, but is required to be >= 5"
            )
        if prediction_length < 1:
            raise ValueError("prediction_length cannot be 0 or negative.")
        if successful_trade_percent <= 0.0:
            raise ValueError(
                "successful_trade_percent cannot be 0 or negative"
            )
        if total_samples <= 0:
            raise ValueError("total_samples cannot be 0 or negative")
        self.ordered_or_shuffled = ordered_or_shuffled
        self.past_window_size = past_window_size
        self.prediction_length = prediction_length
        self.random_dates_total_window = random_dates_total_window
        self.fixed_dates_start = fixed_dates_start
        self.fixed_dates_end = fixed_dates_end
        self.successful_trade_percent = 0.01 * successful_trade_percent
        self.ticker_list_directory = ticker_list_directory
        self.total_samples = total_samples
        self.args = args
        self.kwargs = kwargs.items()
        if random_dates_total_window is None:
            if fixed_dates_end is None and fixed_dates_start is None:
                raise ValueError(
                    "either random_dates_total_window or"
                    "fix_dates input has to have a value"
                )
            if fixed_dates_end is None:
                raise ValueError("fixed_dates_end needs to be filled in")
            if fixed_dates_start is None:
                raise ValueError("fixed_dates_start needs to be filled in")

            end_datetime = datetime.strptime(fixed_dates_end, "%Y-%m-%d")
            start_datetime = datetime.strptime(fixed_dates_start, "%Y-%m-%d")
            if (
                int((end_datetime - start_datetime).days / 7 * 5)
                < max_indicator_length(self.kwargs)
                + past_window_size
                + prediction_length
                + 1
            ):
                raise ValueError(
                    "too small time difference between startdate and enddate."
                )

        elif random_dates_total_window is not None:
            if (
                random_dates_total_window
                < max_indicator_length(self.kwargs)
                + past_window_size
                + prediction_length
                + 1
            ):
                raise ValueError(
                    "total_window must be greater than past_window_size + "
                    "prediction_length + "
                    "the highest window size of the indicators"
                )

    def ticker_list(self):
        """Read the csv file from the class attribute ticker_list_directory.

        and returns a pandas dataframe
        """
        return pd.read_csv(self.ticker_list_directory, header=None)

    def _choose_random_sample(self):
        """Choose valid sample provided by the function ticker_list.

        Examples of invalid samples are:
            1. Samples with error outputs by yfinance library
            2. Samples with Null values
            3. Samples with less data than pass_window_size + prediction_length
        Upon finding invalid samples, the function is run again,
            this time finding another random ticker.
        """
        sample = self.ticker_list().sample().values.flatten()[0]
        if self.random_dates_total_window is not None:
            try:

                ticker = yf.Ticker(sample)
                hist = ticker.history(period="max")
                hist = hist.drop(columns=["Dividends", "Stock Splits"])
                if len(hist.index) < (self.random_dates_total_window + 30):
                    raise RuntimeError(
                        "Too little entries in this current ticker."
                        "Sampling another Ticker..."
                    )
            except Exception:  # pylint: disable=broad-except
                [sample_ticker, sample] = self._choose_random_sample()
            length = len(hist.index)
            if length < self.past_window_size + self.prediction_length:
                [sample_ticker, sample] = self._choose_random_sample()
                return [sample_ticker, sample]

            choices = length - (self.past_window_size + self.prediction_length)
            random_index = random.randint(0, choices)
            return (
                sample,
                hist.iloc[
                    random_index : random_index  # noqa: E203
                    + self.random_dates_total_window
                ],
            )
        try:
            hist = yf.download(
                sample,
                start=self.fixed_dates_start,
                end=self.fixed_dates_end,
            )
            
            print(hist)
            if len(hist.index) == 0:
                raise RuntimeError("Failed Download, resampling...")
            if hist.isnull().values.any():
                raise RuntimeError("NaN values identified. resampling...")
            hist = hist.drop(columns=["Adj Close"])
            return (sample, hist)
        except Exception:  # pylint: disable=broad-except
            [sample_ticker, sample] = self._choose_random_sample()
            return (sample_ticker, sample)

    def _create_df(self, shared_list, counter):
        print("sample number", counter + 1)

        while True:
            sample_ticker, preprocessed_data = self._choose_random_sample()

            try:
                processed_data = custom_arguments(
                    preprocessed_data, self.args, self.kwargs
                )
                normalized_data = normalize_dataframe(processed_data)
                total_columns = len(normalized_data.columns)
                (output_data, shaved_rows) = classify(
                    preprocessed_data,
                    self.prediction_length,
                    self.successful_trade_percent,
                )
                input_data = initial_parameters(
                    normalized_data, total_columns, self.past_window_size
                )
                input_data = input_data[self.past_window_size :]  # noqa: E203
                input_data = input_data[: (len(input_data) - shaved_rows)]
                if len(input_data) < len(output_data):
                    difference = len(output_data) - len(input_data)
                    output_data = output_data[difference:]
                # elif len(output_data) < len(input_data):
                #     input_data = input_data[:(len(output_data))]
                for index, value in enumerate(output_data):
                    input_data[index].append(value)
                shared_list.extend(input_data)
                # print("Write done for process:", counter + 1)
                break
            except ValueError as e:
                print(str(e))
                print("ticker: ", sample_ticker)
                continue

    def generate(self):
        """Generate the training dataset."""
        shared_list = mp.Manager().list()
        processes = []

        start = time.perf_counter()

        for counter in range(self.total_samples):
            process = mp.Process(
                target=self._create_df,
                args=(
                    shared_list,
                    counter,
                ),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        dataframe = pd.DataFrame(list(shared_list))
        dataframe = dataframe.sort_index()

        end = time.perf_counter()

        print("Total time taken: %s" % (end - start))

        if self.ordered_or_shuffled == "shuffled":
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        return dataframe.fillna(0)


if __name__ == "__main__":
    # output_data_frame1 = GenerateTrain(
    #     # "no_open",
    #     # "no_close",
    #     # "no_high",
    #     # "no_volume",
    #     # "no_low",
    #     ordered_or_shuffled="shuffled",
    #     random_dates_total_window=100,
    #     total_samples=1,
    #     adosc={
    #         "primary_columns": ["high", "low", "close", "volume"],
    #         "output_columns": 1,
    #         "period_list": [2, 5],
    #     },
    #     # fixed_dates_start="2017-01-01",
    #     # fixed_dates_end="2017-04-30",
    #     # stoch={
    #     #     "primary_columns": ["high", "low", "close"],
    #     #     "output_columns": 2,
    #     #     "period_list": [2, 3, 5],
    #     # },
    #     # di={
    #     #     "primary_columns": ["high", "low", "close"],
    #     #     "output_columns": 2,
    #     #     "period_list": [5],
    #     # },
    # ).generate()

    output_data_frame2 = GenerateTest(
        # "no_open",
        # "no_close",
        # "no_high",
        # "no_volume",
        # "no_low",
        # adosc={
        #     "primary_columns": ["high", "low", "close", "volume"],
        #     "output_columns": 1,
        #     "period_list": [2, 5],
        # },
        # fixed_dates_start="2017-01-01",
        # fixed_dates_end="2017-04-30",
        # stoch={
        #     "primary_columns": ["high", "low", "close"],
        #     "output_columns": 2,
        #     "period_list": [2, 3, 5],
        # },
        # di={
        #     "primary_columns": ["high", "low", "close"],
        #     "output_columns": 2,
        #     "period_list": [5],
        # },
    ).generate()

    output_data_frame2.to_csv("sampleTrain2.csv", index=False, header=False)
