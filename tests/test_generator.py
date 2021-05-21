import os
import csv
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
    indicator_inputs["fixed_dates_start"] = "2009-10-10"
    indicator_inputs["fixed_dates_end"] = "2010-10-10"
    indicator_inputs["ticker_list_directory"] = ticker_dir
    with tempfile.TemporaryDirectory() as tmpdirname:
        output2 = os.path.join(tmpdirname, "output2.csv")
        GenerateTrain(**indicator_inputs).generate().to_csv(
            output2, index=False, header=False
        )
        with open(output2) as outputfile:
            test_output = list(csv.reader(outputfile))
    assert compare.compare_equal(true_output, test_output)
