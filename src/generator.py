import pandas as pd
import yfinance as yf
import numpy as np
import tulipy as ti
from sklearn import preprocessing
import argparse
import random
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

        past_window_size : how far in the past that you would want the model to have, in order to make its prediction.

        prediction_length : if the model gives a signal, how long in the future does that signal stand?
            (example if model says buy and TP not hit after (prediction_length) intervals, then it's signal is wrong)

        total_window : initial timeframe taken from each sample (prior to shaving).

        successful_trade_percent : take-profit percentage (for both long and short trades).

        total_samples : Number of random samples in the ticker list to sample from.

        ticker_list_directory : file location of the list of tickers.

        **kwargs : Additional commands to add indicators. All indicators can be found at 
        https://tulipindicators.org/list. Currently only support indicator type.

        Add kwargs in the form of:
        YOUR_INDICATOR_NAME={
            "primary_columns": ["high", "low", "close"],    ## primary columns required to compute the indicator
            "output_columns": 2,                            ## number of outputs that the indicator will provide
            "period_list": [2, 3, 5],                       ## add any additional period params in order here
        },

    Outputs:
        The array of lists to be used for training. 


    """

    def __init__(
        self,
        *args,
        past_window_size=5,
        prediction_length=5,
        total_window=100,
        successful_trade_percent=15.0,
        total_samples=20,
        ticker_list_directory="../StockTickers/TickerNames.csv",
        **kwargs
    ):
        if past_window_size < 5:
            raise ValueError(
                "past_window_size is currently < 5, but is required to be >= 5"
            )
        if prediction_length < 1:
            raise ValueError("prediction_length cannot be 0 or negative.")
        if total_window < past_window_size + prediction_length:
            raise ValueError(
                "total_window needs to be greater than past_window_size + prediction_length"
            )
        if successful_trade_percent <= 0.0:
            raise ValueError(
                "successful_trade_percent cannot be 0 or negative")
        if total_samples <= 0:
            raise ValueError("total_samples cannot be 0 or negative")
        self.past_window_size = past_window_size
        self.prediction_length = prediction_length
        self.total_window = total_window
        self.successful_trade_percent = 0.01 * successful_trade_percent
        self.ticker_list_directory = ticker_list_directory
        self.total_samples = total_samples
        self.args = args
        self.kwargs = kwargs.items()

    def _custom_arguments(self, dataframe):
        """
            Adds indicator columns (if any) and remove primary columns (if requested)

            inputs:
                Original dataframe

            outputs:
                Processed dataframe with additional indicator columns and potential primary columns removed.

        """
        dataframe = dataframe.astype("float64")
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

    def ticker_list(self):
        """
            Reads the csv file from the class attribute ticker_list_directory, and returns a pandas dataframe
        """
        return pd.read_csv(self.ticker_list_directory, header=None)

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

    def _choose_sample(self):
        """
            Choose valid sample provided by the function ticker_list.
            Examples of invalid samples are:
                1. Samples with error outputs by yfinance library
                2. Samples with Null values
                3. Samples with less data than pass_window_size + prediction_length
            Upon finding invalid samples, the function is run again, this time finding another random ticker.
        """
        sample = self.ticker_list().sample().values.flatten()[0]
        try:
            ticker = yf.Ticker(sample)
            hist = ticker.history(period="max")
            hist = hist.drop(columns=["Dividends", "Stock Splits"])
            if hist.isnull().values.any():
                raise ValueError("Null Values detected")
        except:
            [sampleTicker, sample] = self._choose_sample()
        length = len(hist.index)
        if length < self.past_window_size + self.prediction_length:
            [sampleTicker, sample] = self._choose_sample()
            return [sampleTicker, sample]
        else:
            choices = length - (self.past_window_size + self.prediction_length)
            randomIndex = random.randint(0, choices)
            return sample, hist.iloc[randomIndex: randomIndex + 100]

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
        for i in range(len(time_series)):
            if i >= self.past_window_size:
                initialize = [0] * (self.past_window_size * column_count)
                for j in range(column_count):
                    for k in range(self.past_window_size):
                        initialize[j * self.past_window_size + k] = time_series.iloc[
                            i - j - 1
                        ][k]
                output_list[i] = initialize
        return output_list

    def generate(self):
        df = pd.DataFrame()
        for _ in range(self.total_samples):
            _, preprocessed_data = self._choose_sample()
            processed_data = self._custom_arguments(preprocessed_data)
            normalized_data = self._normalize_dataframe(processed_data)
            total_columns = len(normalized_data.columns)
            (output_data, shaved_rows) = self._classify(preprocessed_data)
            input_data = self._initial_parameters(
                normalized_data, total_columns)
            input_data = input_data[self.past_window_size:]
            input_data = input_data[: (len(input_data) - shaved_rows - 1)]
            if len(input_data) < len(output_data):
                difference = len(output_data) - len(input_data)
                output_data = output_data[difference:]
            for i in range(len(output_data)):
                input_data[i].append(output_data[i])
            df = pd.concat([df, pd.DataFrame(input_data)], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        return df


if __name__ == "__main__":
    trainingSet = Generator(
        "no_open",
        "no_close",
        stoch={
            "primary_columns": ["high", "low", "close"],
            "output_columns": 2,
            "period_list": [2, 3, 5],
        },
        di={
            "primary_columns": ["high", "low", "close"],
            "output_columns": 2,
            "period_list": [5],
        },
    )

    output_data_frame = trainingSet.generate()
    output_data_frame.to_csv("sampleTrain.csv", index=False, header=False)
