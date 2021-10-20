"""Module containing the generate_all_csv function."""
from tradobot.generator import GenerateTrain
from tradobot.indicator_params import IndicatorDict
from tradobot.neural_network_simple import train_and_test


def generate_all_csv():
    """Generate CSV file for all indicators.

    Then run training and testing on them.
    """
    ind_dic = IndicatorDict()
    indicator_inputs = ind_dic.indicator_dict
    indicator_inputs["past_window_size"] = 5
    indicator_inputs["prediction_length"] = 5
    indicator_inputs["successful_trade_percent"] = 5.0
    indicator_inputs["total_samples"] = 200
    indicator_inputs["ordered_or_shuffled"] = "shuffled"
    indicator_inputs["random_dates_total_window"] = 100
    GenerateTrain(**indicator_inputs).generate().to_csv(
        "./csv_files/EVERYTHING.csv", index=False, header=False
    )

    train_accuracy, val_accuracy, model = train_and_test("./csv_files/EVERYTHING.csv")
    model.save("./model")
    print(train_accuracy, val_accuracy)


if __name__ == "__main__":
    generate_all_csv()
