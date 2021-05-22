"""Module containing the generate_all_csv function."""
from tradobot.generator import GenerateTrain
from tradobot.indicator_params import IndicatorDict


def generate_all_csv():
    """Generate CSV file for all indicators."""
    ind_dic = IndicatorDict()
    indicator_inputs = ind_dic.indicator_dict
    indicator_inputs["past_window_size"] = 5
    indicator_inputs["prediction_length"] = 5
    indicator_inputs["successful_trade_percent"] = 5.0
    indicator_inputs["total_samples"] = 1
    indicator_inputs["ordered_or_shuffled"] = "ordered"
    indicator_inputs["fixed_dates_start"] = "2007-10-10"
    indicator_inputs["fixed_dates_end"] = "2008-10-10"
    indicator_inputs[
        "ticker_list_directory"
    ] = "../../fixtures/aapl/TickerNames.csv"
    GenerateTrain(**indicator_inputs).generate().to_csv(
        "./output.csv", index=False, header=False
    )


if __name__ == "__main__":
    generate_all_csv()
