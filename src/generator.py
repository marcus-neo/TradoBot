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

    def total_columns(self):
        return 5

    def _custom_arguments(self, dataframe):
        dataframe = dataframe.astype('float64')
        open_list = dataframe["Open"].to_numpy()
        high_list = dataframe["High"].to_numpy()
        low_list = dataframe["Low"].to_numpy()
        close_list = dataframe["Close"].to_numpy()
        volume_list = dataframe["Volume"].to_numpy()

        for key, value in self.kwargs:
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
            class_method = getattr(ti, key)
            print(class_method)
            print(param)
            custom_output = eval("class_method" + "(" + param + ")")
            if custom_output.ndim == 1:
                dataframe[class_method.__name__] = custom_output
            else:
                for i in range(value[1]):
                    dataframe[class_method.__name__ +
                              str(i)] = custom_output[i]

        return dataframe

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

    def _initial_parameters(self, time_series):
        columns = self.total_columns()
        output_list = [([0] * (self.past_window_size * columns))
                       ] * len(time_series)
        for i in range(len(time_series)):
            if i >= self.past_window_size:
                initialize = [0] * (self.past_window_size * columns)
                for j in range(self.past_window_size):
                    for k in range(columns):
                        initialize[j * self.past_window_size +
                                   k] = time_series.iloc[i - j - 1][k]
                output_list[i] = initialize
        return output_list

    def generate(self):
        df = pd.DataFrame()
        for _ in range(self.total_samples):
            sample_name, preprocessed_data = self._choose_sample()
            # print(sample)
            # row_names = list(sample.index)
            # dates = [d.strftime("%m-%d-%Y") for d in row_names]
            processed_data = self._custom_arguments(preprocessed_data)
            normalized_data = self._normalize_dataframe(processed_data)
            print(normalized_data)
            output_data = self._classify(preprocessed_data)
            input_data = self._initial_parameters(preprocessed_data)
            input_data = input_data[self.past_window_size:]
            input_data = input_data[:len(output_data)]
            for i in range(len(output_data)):
                input_data[i].append(output_data[i])
            df = pd.concat([df, pd.DataFrame(input_data)], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        return df


if __name__ == "__main__":
    trainingSet = Generator(ad=(["high", "low", "close", "volume"], 1))
    output_data_frame = trainingSet.generate()
    # print(output_data_frame)
    output_data_frame.to_csv("sampleTrain.csv")
