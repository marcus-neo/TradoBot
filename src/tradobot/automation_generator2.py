from tradobot.generator import GenerateTrain
from tradobot.indicator_params import indicator_dict
from tradobot.neural_network_simple import train_and_test
import pandas as pd


def generate_all_csv():
    ind_dic = indicator_dict()
    indicator_inputs = ind_dic.indicator_dict
    indicator_inputs["past_window_size"] = 5
    indicator_inputs["prediction_length"] = 5
    indicator_inputs["successful_trade_percent"] = 5.0
    indicator_inputs["total_samples"] = 1
    indicator_inputs["ordered_or_shuffled"] = "ordered"
    indicator_inputs["fixed_dates_start"] = "2009-10-10"
    indicator_inputs["fixed_dates_end"] = "2010-10-10"
    indicator_inputs["ticker_list_directory"] = "../fixtures/aapl/TickerNames.csv"
    GenerateTrain(**indicator_inputs).generate().to_csv(
        "./output.csv", index=False, header=False)

    # train_accuracy, val_accuracy = train_and_test(
    #     "./output.csv")
    # print(train_accuracy, val_accuracy)
    # with open(indicators_filename, "r") as fd:
    #     lines=fd.readlines()
    #     for line in lines:
    #         indicator_inputs={}
    #         indicator_lst=ast.literal_eval(line)
    #         print(indicator_lst)
    #         for indicator in indicator_lst:
    #             indicator_inputs[indicator]=ind_dic.get_params(indicator)
    #         indicator_inputs["ordered_or_shuffled"]="shuffled"
    #         indicator_inputs["random_dates_total_window"]=100
    #         try:
    #             Generator(**indicator_inputs).generate().to_csv(
    #                 "./csv_files/" + str(indicator_lst) + ".csv", index = False, header = False)
    #             train_accuracy, val_accuracy=train_and_test("./csv_files/" + str(indicator_lst) +
    #                                                           ".csv")
    #             with open("./log/log.txt", "a+") as logfile:
    #                 logfile.write(str(indicator_lst) + " : " +
    #                               str((train_accuracy, val_accuracy)) + "\n")
    #         except ValueError:
    #             with open("./log/log.txt", "a+") as logfile:
    #                 logfile.write(str(indicator_lst) + " : ERROR\n")


if __name__ == "__main__":
    generate_all_csv()
