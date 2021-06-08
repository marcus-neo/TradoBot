from tradobot.generator import GenerateTest
from tradobot.indicator_params import IndicatorDict
def ind_inp():
    ind_dic = IndicatorDict()
    indicator_inputs = ind_dic.indicator_dict
    indicator_inputs["past_window_size"] = 5
    indicator_inputs["prediction_length"] = 5
    indicator_inputs["successful_trade_percent"] = 5.0
    indicator_inputs["total_samples"] = 200
    indicator_inputs["total_window"] = 100
    return indicator_inputs

ind_inputs = ind_inp()

def ind_pred(ticker_name):
    ind_inputs["ticker_name"] = ticker_name
    output_lst = GenerateTest(
        **ind_inputs
    ).generate().values.tolist()
    return (
        output_lst
        if output_lst == []
        else output_lst[0]
    )
