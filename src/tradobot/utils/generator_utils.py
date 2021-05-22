import numpy as np
import tulipy as ti
import pandas as pd
from sklearn import preprocessing

def normalize_dataframe(dataframe):
    """Normalize all columns in the dataframe.

    (normalised with respect to only data within the column)

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


def custom_arguments(dataframe, given_args, given_kwargs):
    """Add indicator columns (if any).

    and remove primary columns (if requested)..
    inputs:
        Original dataframe

    outputs:
        Processed dataframe with additional indicator columns and
        potential primary columns removed.
    """
    dataframe = dataframe.astype("float64")
    # pylint: disable=unused-variable
    open_list = dataframe[  # noqa: F841
        "Open"
    ].to_numpy()  # pylint: disable=unused-variable
    high_list = dataframe[  # noqa: F841
        "High"
    ].to_numpy()  # pylint: disable=unused-variable
    low_list = dataframe[  # noqa: F841
        "Low"
    ].to_numpy()  # pylint: disable=unused-variable
    close_list = dataframe[  # noqa: F841
        "Close"
    ].to_numpy()  # pylint: disable=unused-variable
    volume_list = dataframe[  # noqa: F841
        "Volume"
    ].to_numpy()  # pylint: disable=unused-variable
    smallest_output = len(dataframe)
    extra_column_dict = {}
    if (
        all(
            item in given_args
            for item in [
                "no_open",
                "no_close",
                "no_high",
                "no_low",
                "no_volume",
            ]
        )
    ) and not bool(given_kwargs):
        raise ValueError(
            "With all these input parameters, "
            "there will be no nothing to evaluate on. \n"
            "Either add indicators or do not remove all primary columns."
        )
    for item in given_args:
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

    for key, value in given_kwargs:
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
        key = "".join(ch for ch in key if not ch.isupper())
        class_method = getattr(ti, key)
        # pylint: disable=eval-used
        custom_output = eval("class_method" + "(" + param + ")")
        if not isinstance(custom_output, tuple):
            if len(custom_output) < len(dataframe):
                smallest_output = min(smallest_output, len(custom_output))
            extra_column_dict[class_method.__name__] = custom_output
        else:
            for i in range(value["output_columns"]):
                if len(custom_output[i]) < len(dataframe):
                    smallest_output = min(
                        smallest_output, len(custom_output[i])
                    )
                extra_column_dict[
                    class_method.__name__ + str(i)
                ] = custom_output[i]
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


def classify(preprocessed_data, prediction_length, successsful_trade_percent):
    """Generate output classifiers for individual data."""
    close = preprocessed_data["Close"].to_numpy()
    high = preprocessed_data["High"].to_numpy()
    low = preprocessed_data["Low"].to_numpy()
    output = [0] * len(close)
    for index, _ in enumerate(close):
        if index >= prediction_length:
            offset = 1
            position = 0
            while (
                index + offset < len(close) - 1
                and position == 0
                and offset <= prediction_length
            ):
                offset += 1
                if (
                    low[index + offset]
                    <= (1 - successsful_trade_percent) * close[index]
                ):
                    position = -1
                    break
                if (
                    high[index + offset]
                    >= (1 + successsful_trade_percent) * close[index]
                ):
                    position = 1
                    break
            output[index] = position
    pos = len(output) - 1
    for index, _ in enumerate(output):
        if output[-index - 1] != 0:
            break
        pos -= 1

    shaved_output = output[prediction_length:pos]  # noqa: E203
    return (
        shaved_output,
        (
            len(preprocessed_data.index)
            - len(shaved_output)
            - prediction_length
        ),
    )


def initial_parameters(time_series, column_count, past_window_size):
    """Create initial parameters."""
    output_list = [([0] * (past_window_size * column_count))] * len(
        time_series
    )
    for current_ts_position in range(len(time_series)):
        if current_ts_position >= past_window_size:
            initialize = [0] * (past_window_size * column_count)
            for column_index in range(column_count):
                for t_minus_n in range(past_window_size):
                    initialize[
                        column_index * past_window_size + t_minus_n
                    ] = time_series.iloc[current_ts_position - t_minus_n - 1][
                        column_index
                    ]
            output_list[current_ts_position] = initialize
    return output_list


def max_indicator_length(given_kwargs):
    """Obtain the max indicator look-back from all indicators used."""
    max_length = 0
    for _, indicator_params in given_kwargs:
        if "period_list" in indicator_params:
            max_length = max(max_length, max(indicator_params["period_list"]))
    return max_length