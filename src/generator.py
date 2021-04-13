import pandas as pd
import yfinance as yf
import numpy as np
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
        self.kwargs = kwargs or None

    def total_columns(self):
        return 5

    def ticker_list(self):
        return pd.read_csv(self.ticker_list_directory, header=None)

    def _choose_sample(self):
        def __normalize_dataframe(dataframe):
            vals = dataframe
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled_vals = min_max_scaler.fit_transform(vals)
            dataframe = pd.DataFrame(
                scaled_vals, columns=dataframe.columns, index=dataframe.index)
            return dataframe

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
            return sample, __normalize_dataframe(hist.iloc[randomIndex: randomIndex + 100])

    def _classify(self, sample):
        close = sample["Close"].to_numpy()
        high = sample["High"].to_numpy()
        low = sample["Low"].to_numpy()
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
            sample_name, sample = self._choose_sample()
            # print(sample)
            row_names = list(sample.index)
            dates = [d.strftime("%m-%d-%Y") for d in row_names]
            output_data = self._classify(sample)
            input_data = self._initial_parameters(sample)
            input_data = input_data[self.past_window_size:]
            input_data = input_data[:len(output_data)]
            for i in range(len(output_data)):
                input_data[i].append(output_data[i])
            df = pd.concat([df, pd.DataFrame(input_data)], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        return df


if __name__ == "__main__":
    trainingSet = Generator()
    output_data_frame = trainingSet.generate()
    print(output_data_frame)
    output_data_frame.to_csv("sampleTrain.csv")
