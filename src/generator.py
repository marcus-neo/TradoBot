import pandas as pd
import yfinance as yf
import numpy as np
import tulipy as ti
from sklearn import preprocessing
# import keras
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
            windows discarded by indicators -
            windows discarded by



    """

    def __init__(
        self,
        past_window_size=5,
        prediction_length=5,
        total_window=100,
        successful_trade_percent=15.,
        total_samples=1,
        ticker_list_directory="../StockTickers/TickerNames2.csv",
        **kwargs
    ):
        if past_window_size < 5:
            raise ValueError(
                "past_window_size is currently < 5, but is required to be >= 5"
            )
        if prediction_length < 1:
            raise ValueError(
                "prediction_length cannot be 0 or negative."
            )
        if total_window < past_window_size + prediction_length:
            raise ValueError(
                "total_window needs to be greater than past_window_size + prediction_length"
            )
        if successful_trade_percent <= 0.:
            raise ValueError(
                "successful_trade_percent cannot be 0 or negative"
            )
        if total_samples <= 0:
            raise ValueError(
                "total_samples cannot be 0 or negative"
            )
        self.past_window_size = past_window_size
        self.prediction_length = prediction_length
        self.total_window = total_window
        self.successful_trade_percent = 0.01*successful_trade_percent
        self.ticker_list_directory = ticker_list_directory
        self.total_samples = total_samples
        self.kwargs = kwargs.items()

    def _custom_arguments(self, dataframe):
        dataframe = dataframe.astype('float64')
        open_list = dataframe["Open"].to_numpy()
        high_list = dataframe["High"].to_numpy()
        low_list = dataframe["Low"].to_numpy()
        close_list = dataframe["Close"].to_numpy()
        volume_list = dataframe["Volume"].to_numpy()
        smallest_output = len(dataframe)
        extra_column_dict = {}
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
                    param += (", " + str(value["period_list"][i]))

            class_method = getattr(ti, key)
            custom_output = eval("class_method" + "(" + param + ")")
            if not isinstance(custom_output, tuple):
                if len(custom_output) < len(dataframe):
                    print(len(custom_output))
                    smallest_output = min(smallest_output, len(custom_output))
                    # difference = len(dataframe) - len(custom_output)
                    # dataframe = dataframe.iloc[difference:]
                extra_column_dict[class_method.__name__] = custom_output
                # dataframe[class_method.__name__] = custom_output
            else:
                # difference = len(dataframe) - len(custom_output[0])
                # dataframe = dataframe.iloc[difference:]

                for i in range(value["output_columns"]):
                    if len(custom_output[i]) < len(dataframe):
                        smallest_output = min(
                            smallest_output, len(custom_output[i]))
                    extra_column_dict[class_method.__name__ +
                                      str(i)] = custom_output[i]
                    # dataframe[class_method.__name__ +
                    #             str(i)] = custom_output[i]
        difference = len(dataframe) - smallest_output
        dataframe = dataframe.iloc[difference:]
        with pd.option_context('mode.chained_assignment', None):
            for key, value in extra_column_dict.items():
                new_difference = len(value)-smallest_output
                if new_difference > 0:
                    new_value = np.delete(value, slice(0, new_difference), 0)
                    dataframe[key] = new_value
                else:
                    dataframe[key] = value
        return dataframe, smallest_output

    def ticker_list(self):
        return pd.read_csv(self.ticker_list_directory, header=None)

    def _normalize_dataframe(self, dataframe):
        vals = dataframe
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_vals = min_max_scaler.fit_transform(vals)
        dataframe = pd.DataFrame(
            scaled_vals, columns=dataframe.columns, index=dataframe.index)
        return dataframe

    def _choose_sample(self):

        sample = self.ticker_list().sample().values.flatten()[0]
        try:
            ticker = yf.Ticker(sample)
            hist = ticker.history(period="3Y")
        except:
            [sampleTicker, sample] = self._chooseSample()
        hist = hist.drop(columns=["Dividends", "Stock Splits"])
        length = len(hist.index)
        if length < self.past_window_size + self.prediction_length:
            [sampleTicker, sample] = self._chooseSample()
            return [sampleTicker, sample]
        else:
            choices = length - 100
            randomIndex = random.randint(0, choices)
            return sample, hist.iloc[randomIndex: randomIndex + 100]

    def _classify(self, preprocessed_data):
        close = preprocessed_data["Close"].to_numpy()
        high = preprocessed_data["High"].to_numpy()
        low = preprocessed_data["Low"].to_numpy()
        output = [0] * len(close)
        for i in range(len(close)):
            if i >= self.prediction_length:
                base_price = close[i]
                offset = 1
                position = 0
                while i + offset < len(close) - 1 and position == 0 and offset <= self.prediction_length:
                    offset += 1
                    if low[i+offset] <= (1-self.successful_trade_percent) * close[i]:
                        position = -1
                        break
                    elif high[i+offset] >= (1+self.successful_trade_percent) * close[i]:
                        position = 1
                        break
                output[i] = position
        pos = len(output) - 1
        for i in range(len(output)):
            if output[-i-1] != 0:
                break
            else:
                pos -= 1
        return output[self.prediction_length:pos-1]

    def _initial_parameters(self, time_series, column_count):
        output_list = [([0] * (self.past_window_size * column_count))
                       ] * len(time_series)
        for i in range(len(time_series)):
            if i >= self.past_window_size:
                initialize = [0] * (self.past_window_size * column_count)
                for j in range(column_count):
                    for k in range(self.past_window_size):
                        initialize[j * self.past_window_size +
                                   k] = time_series.iloc[i - j - 1][k]
                output_list[i] = initialize
        return output_list

    def generate(self):
        df = pd.DataFrame()
        for _ in range(self.total_samples):
            sample_name, preprocessed_data = self._choose_sample()
            processed_data, df_len = self._custom_arguments(preprocessed_data)
            normalized_data = self._normalize_dataframe(processed_data)
            total_columns = len(normalized_data.columns)
            output_data = self._classify(preprocessed_data)
            input_data = self._initial_parameters(
                normalized_data, total_columns)
            input_data = input_data[self.past_window_size:]
            if len(input_data) > len(output_data):
                input_data = input_data[: len(output_data)]
            elif len(input_data) < len(output_data):
                output_data = output_data[: len(input_data)]
            for i in range(len(output_data)):
                input_data[i].append(output_data[i])
            df = pd.concat([df, pd.DataFrame(input_data)], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        return df


if __name__ == "__main__":
    trainingSet = Generator(
        stoch={
            "primary_columns": ["high", "low", "close"],
            "output_columns": 2,
            "period_list": [2, 3, 5]
        },
        di={
            "primary_columns": ["high", "low", "close"],
            "output_columns": 2,
            "period_list": [5]
        }
    )

    output_data_frame = trainingSet.generate()
    output_data_frame.to_csv("sampleTrain.csv")
