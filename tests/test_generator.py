import os
import csv
import datetime
import tempfile
import compare

from tradobot.indicator_params import IndicatorDict
from tradobot.generator import GenerateTrain


def test_generator(aapl_test):

    (ticker_dir, true_output) = aapl_test
    ind_dic = IndicatorDict()
    indicator_inputs = ind_dic.indicator_dict
    indicator_inputs["past_window_size"] = 5
    indicator_inputs["prediction_length"] = 5
    indicator_inputs["successful_trade_percent"] = 5.0
    indicator_inputs["total_samples"] = 1
    indicator_inputs["ordered_or_shuffled"] = "ordered"
    indicator_inputs["fixed_dates_start"] = (
        datetime.datetime.fromisoformat(
            '2007-10-10T00:00:00-04:00'
        ).date().strftime("%Y-%m-%d")
    )
    indicator_inputs["fixed_dates_end"] = (
        datetime.datetime.fromisoformat(
            '2008-10-10T00:00:00-04:00'
        ).date().strftime("%Y-%m-%d")
    )
    indicator_inputs["ticker_list_directory"] = ticker_dir
    with tempfile.TemporaryDirectory() as tmpdirname:
        output2 = os.path.join(tmpdirname, "output2.csv")
        generator_output = GenerateTrain(**indicator_inputs).generate()
        generator_output.to_csv(
            output2, index=False, header=False
        )
        with open(output2) as outputfile:
            test_output = list(csv.reader(outputfile))
    assert compare.compare_equal(true_output, test_output)
