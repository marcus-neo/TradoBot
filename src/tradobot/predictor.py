import os
from multiprocessing import Pool
from datetime import date, datetime
import pytz
import pandas as pd
import tensorflow as tf
from tradobot.utils.predictor_utils import ind_pred

def predict(ticker_directory: str, model_directory: str):
    tickerfile = os.path.join(ticker_directory)
    dataframe = pd.read_csv(tickerfile)
    unflat_lst = dataframe.values.tolist()
    lst = [item[0] for item in unflat_lst]

    model = tf.saved_model.load(os.path.join(model_directory))
    pool = Pool()
    input_list = pool.map(ind_pred, lst)
    pool.close()
    pool.join()
    failed_indices = [
        index
        for index, item in enumerate(input_list)
        if item == []
    ]
    input_list = [
        item
        for index, item in enumerate(input_list)
        if index not in failed_indices
    ]
    lst = [
        item
        for index, item in enumerate(lst)
        if index not in failed_indices
    ]
    input_tensor = tf.convert_to_tensor(input_list, dtype=tf.float32)
    output_list = model(input_tensor).numpy().tolist()
    choices = [
        [(index, score)
        for index, score in enumerate(output) if score == max(output)]
        for output in output_list
    ]
    choices_dict = {
        key:[value[0][0], value[0][1]]
        for key, value in zip(lst, choices)
    }
    timez = pytz.timezone("America/Cancun")
    currtime = datetime.now(tz=timez).date()
    output_pd = pd.DataFrame.from_dict(
        choices_dict, orient="index", columns=["Choice", "Confidence"]
    )
    output_pd = output_pd.sort_values(by="Confidence", ascending=False)
    output_pd.to_csv(
        "./results/" + currtime.strftime("%d%m%y") + ".csv"
    )


if __name__ == "__main__":
    predict("./tradobot/StockTickers/TickerNames.csv", "./model")
