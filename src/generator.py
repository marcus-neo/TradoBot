import pandas as pd
import yfinance as yf
import numpy as np
import tulipy as ti
from sklearn import preprocessing
import random
from datetime import date, timedelta
import multiprocessing as mp
import time
import datetime


class Generator:
    """
    Creates an array of lists
        The list size is determined by:
            total indicator outputs (rsi, etc) +
            total raw outputs (open, high, low, close, vol)
        The array size is determined by:
            total_window -
            windows discarded by past_window_size considerations -
            windows discarded by indicators -
            windows discarded by zero-tail outputs
            (if last few rows are zeros, they'll be discarded until a non-zero output is found.
            This is to increase the number of zero outputs as they are far too common)
    Inputs:
        *args : Additional commands to remove primary rows from output.
            Possible commands are "no_open", "no_close", "no_high", "no_low", "no_volume"

        ordered_or_shuffled : Whether the data is ordered or randomly shuffled
        past_window_size : how far in the past that you would want the model to have, in order to make its prediction.

        prediction_length : if the model gives a signal, how long in the future does that signal stand?
            (example if model says buy and TP not hit after (prediction_length) intervals, then it's signal is wrong)

        random_dates_total_window : The window that will be used for each sample, ending with the last available trading date.

        fixed_dates_start : The start date

        fixed_dates_end : The end date

        successful_trade_percent : take-profit percentage (for both long and short trades).

        total_samples : Number of random samples in the ticker list to sample from.

        ticker_list_directory : file location of the list of tickers.

        **kwargs : Additional commands to add indicators. All indicators can be found at
        https://tulipindicators.org/list. Currently only support indicator type.

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

    def _custom_arguments(self, dataframe):
        """
        Adds indicator columns (if any) and remove primary columns (if requested)

        inputs:
            Original dataframe

        outputs:
            Processed dataframe with additional indicator columns and potential primary columns removed.

        """
        dataframe = dataframe.astype("float64")
        # pylint: disable=unused-variable
        open_list = dataframe["Open"].to_numpy(
        )  # pylint: disable=unused-variable
        high_list = dataframe["High"].to_numpy(
        )  # pylint: disable=unused-variable
        low_list = dataframe["Low"].to_numpy(
        )  # pylint: disable=unused-variable
        close_list = dataframe["Close"].to_numpy(
        )  # pylint: disable=unused-variable
        volume_list = dataframe["Volume"].to_numpy(
        )  # pylint: disable=unused-variable
        smallest_output = len(dataframe)
        extra_column_dict = {}
        if (
            all(
                item in self.args
                for item in ["no_open", "no_close", "no_high", "no_low", "no_volume"]
            )
        ) and not bool(self.kwargs):
            raise ValueError(
                "With all these input parameters, there will be no nothing to evaluate on. \n Either add indicators or do not remove all primary columns."
            )
        for item in self.args:
            if item == "no_open":
                dataframe = dataframe.drop(columns="Open")
            elif item == "no_close":
                dataframe = dataframe.drop(columns="Close")
            elif item == "no_high":
                dataframe = dataframe.drop(columns="High")
            elif item == "no_low":
                dataframe = dataframe.drop(columns="Low")
            elif item == "no_volume":
                dataframe = dataframe.drop(columns="Volume")
            else:
                raise ValueError("Invalid argument " + item)

        for key, value in self.kwargs:
            param = ""
            if "open" in value["primary_columns"]:
                param += "open_list"
            if "high" in value["primary_columns"]:
                if param:
                    param += ", "
                param += "high_list"
            if "low" in value["primary_columns"]:
                if param:
                    param += ", "
                param += "low_list"
            if "close" in value["primary_columns"]:
                if param:
                    param += ", "
                param += "close_list"
            if "volume" in value["primary_columns"]:
                if param:
                    param += ", "
                param += "volume_list"
            if "period_list" in value:
                for i in range(len(value["period_list"])):
                    param += ", " + str(value["period_list"][i])
            key = ''.join(ch for ch in key if not ch.isupper())
            class_method = getattr(ti, key)
            custom_output = eval("class_method" + "(" + param + ")")
            if not isinstance(custom_output, tuple):
                if len(custom_output) < len(dataframe):
                    smallest_output = min(smallest_output, len(custom_output))
                extra_column_dict[class_method.__name__] = custom_output
            else:
                for i in range(value["output_columns"]):
                    if len(custom_output[i]) < len(dataframe):
                        smallest_output = min(
                            smallest_output, len(custom_output[i]))
                    extra_column_dict[class_method.__name__ +
                                      str(i)] = custom_output[i]
        difference = len(dataframe) - smallest_output
        dataframe = dataframe.iloc[difference:]
        with pd.option_context("mode.chained_assignment", None):
            for key, value in extra_column_dict.items():
                new_difference = len(value) - smallest_output
                if new_difference > 0:
                    new_value = np.delete(value, slice(0, new_difference), 0)
                    dataframe[key] = new_value
                else:
                    dataframe[key] = value
        return dataframe

    def _normalize_dataframe(self, dataframe):
        """
        Normalizes all columns in the dataframe (normalised with respect to only data within the column)

        Input:
            Unnormalized dataframe

        Output:
            Normalized dataframe
        """
        vals = dataframe
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_vals = min_max_scaler.fit_transform(vals)
        dataframe = pd.DataFrame(
            scaled_vals, columns=dataframe.columns, index=dataframe.index
        )
        return dataframe

    def _classify(self, preprocessed_data):
        """
        Generate output classifiers for individual data.
        """
        close = preprocessed_data["Close"].to_numpy()
        high = preprocessed_data["High"].to_numpy()
        low = preprocessed_data["Low"].to_numpy()
        output = [0] * len(close)
        for i in range(len(close)):
            if i >= self.prediction_length:
                offset = 1
                position = 0
                while (
                    i + offset < len(close) - 1
                    and position == 0
                    and offset <= self.prediction_length
                ):
                    offset += 1
                    if (
                        low[i + offset]
                        <= (1 - self.successful_trade_percent) * close[i]
                    ):
                        position = -1
                        break
                    elif (
                        high[i + offset]
                        >= (1 + self.successful_trade_percent) * close[i]
                    ):
                        position = 1
                        break
                output[i] = position
        pos = len(output) - 1
        for i in range(len(output)):
            if output[-i - 1] != 0:
                break
            else:
                pos -= 1

        shaved_output = output[self.prediction_length: pos]
        return (
            shaved_output,
            (
                len(preprocessed_data.index)
                - len(shaved_output)
                - self.prediction_length
            ),
        )

    def _initial_parameters(self, time_series, column_count):
        output_list = [([0] * (self.past_window_size * column_count))] * len(
            time_series
        )
        for current_TS_position in range(len(time_series)):
            if current_TS_position >= self.past_window_size:
                initialize = [0] * (self.past_window_size * column_count)
                for column_index in range(column_count):
                    for t_minus_n in range(self.past_window_size):
                        initialize[
                            column_index * self.past_window_size + t_minus_n
                        ] = time_series.iloc[current_TS_position - t_minus_n - 1][
                            column_index
                        ]
                output_list[current_TS_position] = initialize
        return output_list

    def max_indicator_length(self):
        max_length = 0
        for _, indicator_params in self.kwargs:
            if "period_list" in indicator_params:
                max_length = max(max_length, max(
                    indicator_params["period_list"]))
        return max_length


class GenerateTest(Generator):
    def __init__(
        self,
        *args,
        past_window_size=5,
        prediction_length=5,
        successful_trade_percent=5.0,
        total_window_length=100,
        ticker_name="AAPL",
        **kwargs
    ):
        # Error Catching
        if past_window_size < 5:
            raise ValueError(
                "past_window_size is currently < 5, but is required to be >= 5"
            )
        if prediction_length < 1:
            raise ValueError("prediction_length cannot be 0 or negative.")
        if successful_trade_percent <= 0.0:
            raise ValueError(
                "successful_trade_percent cannot be 0 or negative")
        self.past_window_size = past_window_size
        self.prediction_length = prediction_length
        self.successful_trade_percent = 0.01 * successful_trade_percent
        self.ticker_name = ticker_name
        self.args = args
        self.kwargs = kwargs.items()
        if total_window_length < past_window_size + self.max_indicator_length() + 1:
            raise ValueError(
                "total_window_length too small"
            )
        self.total_window = total_window_length

    def _choose_sample(self):
        start_date = date.today()-timedelta(days=int(self.total_window/5*7))
        hist = yf.download(
            self.ticker_name, start=start_date, end=date.today()
        )
        if len(hist.index) == 0:
            raise RuntimeError("Failed Download, empty dataset")
        if hist.isnull().values.any():
            raise RuntimeError("Failed Download, NaN values identified.")
        hist = hist.drop(columns=["Adj Close"])
        return (self.ticker_name, hist)

    def _create_df(self, preprocessed_data, df):
        processed_data = self._custom_arguments(preprocessed_data)
        normalized_data = self._normalize_dataframe(processed_data)
        total_columns = len(normalized_data.columns)
        input_data = self._initial_parameters(
            normalized_data, total_columns)
        input_data = input_data[self.past_window_size:]
        print(input_data[-1])
        df = pd.concat([df, pd.DataFrame(input_data)]).tail(1)
        df = df.sort_index()
        return df

    def generate(self):
        df = pd.DataFrame()
        ticker, preprocessed_data = self._choose_sample()
        try:
            df = self._create_df(preprocessed_data, df)
        except ValueError as e:
            print(str(e))
            print("ticker: ", ticker)
        return df.fillna(0)


class GenerateTrain(Generator):
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
        # Error Catching
        if ordered_or_shuffled not in ["shuffled", "ordered"]:
            raise ValueError(
                "invalid entry 'ordered_or_shuffled' parameter. Only 'ordered' or 'shuffled' are accepted."
            )
        if past_window_size < 5:
            raise ValueError(
                "past_window_size is currently < 5, but is required to be >= 5"
            )
        if prediction_length < 1:
            raise ValueError("prediction_length cannot be 0 or negative.")
        if successful_trade_percent <= 0.0:
            raise ValueError(
                "successful_trade_percent cannot be 0 or negative")
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
            if fixed_dates_end == None and fixed_dates_start == None:
                raise ValueError(
                    "either random_dates_total_window or fix_dates input has to have a value"
                )
            elif fixed_dates_end == None:
                raise ValueError("fixed_dates_end needs to be filled in")
            elif fixed_dates_start == None:
                raise ValueError("fixed_dates_start needs to be filled in")
            else:
                end_datetime = datetime.strptime(fixed_dates_end, "%Y-%m-%d")
                start_datetime = datetime.strptime(
                    fixed_dates_start, "%Y-%m-%d")
                if int((end_datetime-start_datetime).days/7*5) < self.max_indicator_length + past_window_size + prediction_length + 1:
                    raise ValueError(
                        "too small time difference between startdate and enddate."
                    )

        elif not random_dates_total_window is None:
            if random_dates_total_window < self.max_indicator_length() + past_window_size + prediction_length + 1:
                raise ValueError(
                    "total_window needs to be greater than past_window_size + prediction_length + the highest window size of the indicators"
                )

    def ticker_list(self):
        """
        Reads the csv file from the class attribute ticker_list_directory, and returns a pandas dataframe
        """
        return pd.read_csv(self.ticker_list_directory, header=None)

    def _choose_random_sample(self):
        """
        Choose valid sample provided by the function ticker_list.
        Examples of invalid samples are:
            1. Samples with error outputs by yfinance library
            2. Samples with Null values
            3. Samples with less data than pass_window_size + prediction_length
        Upon finding invalid samples, the function is run again, this time finding another random ticker.
        """
        sample = self.ticker_list().sample().values.flatten()[0]
        if self.random_dates_total_window is not None:
            try:

                ticker = yf.Ticker(sample)
                hist = ticker.history(period="max")
                hist = hist.drop(columns=["Dividends", "Stock Splits"])
                if len(hist.index) < (self.random_dates_total_window + 30):
                    raise RuntimeError(
                        "Too little entries in this current ticker. Sampling another Ticker..."
                    )
            except:
                [sampleTicker, sample] = self._choose_random_sample()
            length = len(hist.index)
            if length < self.past_window_size + self.prediction_length:
                [sampleTicker, sample] = self._choose_random_sample()
                return [sampleTicker, sample]
            else:
                choices = length - (self.past_window_size +
                                    self.prediction_length)
                randomIndex = random.randint(0, choices)
                return (
                    sample,
                    hist.iloc[
                        randomIndex: randomIndex + self.random_dates_total_window
                    ],
                )
        else:
            try:
                hist = yf.download(
                    sample, start=self.fixed_dates_start, end=self.fixed_dates_end
                )
                if len(hist.index) == 0:
                    raise RuntimeError("Failed Download, resampling...")
                if hist.isnull().values.any():
                    raise RuntimeError("NaN values identified. resampling...")
                hist = hist.drop(columns=["Adj Close"])
                return (sample, hist)
            except:
                [sampleTicker, sample] = self._choose_random_sample()
                return (sampleTicker, sample)

    def _create_df(self, shared_list, counter):
        print("sample number", counter + 1)

        while True:
            sample_ticker, preprocessed_data = self._choose_random_sample()

            try:
                processed_data = self._custom_arguments(preprocessed_data)
                normalized_data = self._normalize_dataframe(processed_data)
                total_columns = len(normalized_data.columns)
                (output_data, shaved_rows) = self._classify(preprocessed_data)
                input_data = self._initial_parameters(
                    normalized_data, total_columns)
                input_data = input_data[self.past_window_size:]
                input_data = input_data[: (len(input_data) - shaved_rows)]
                if len(input_data) < len(output_data):
                    difference = len(output_data) - len(input_data)
                    output_data = output_data[difference:]
                # elif len(output_data) < len(input_data):
                #     input_data = input_data[:(len(output_data))]
                for i in range(len(output_data)):
                    input_data[i].append(output_data[i])
                shared_list.extend(input_data)
                # print("Write done for process:", counter + 1)
                break
            except ValueError as e:
                print(str(e))
                print("ticker: ", sample_ticker)
                continue

    def generate(self):
        shared_list = mp.Manager().list()
        processes = []

        start = time.perf_counter()

        for counter in range(self.total_samples):
            process = mp.Process(target=self._create_df,
                                 args=(shared_list, counter,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        df = pd.DataFrame(list(shared_list))
        df = df.sort_index()

        end = time.perf_counter()

        print("Total time taken: %s" % (end - start))

        if self.ordered_or_shuffled == "shuffled":
            df = df.sample(frac=1).reset_index(drop=True)
            return df.fillna(0)
        elif self.ordered_or_shuffled == "ordered":
            return df.fillna(0)
        else:
            raise RuntimeError(
                "Supposed to be caught in init but... 'ordered_or_shuffled' must only contain 'ordered' or 'shuffled'"
            )


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
