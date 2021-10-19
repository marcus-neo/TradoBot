from itertools import chain, combinations
import ast
import pickle


class indicator_dict:
    def __init__(self):
        self.indicator_dict = [
            {
                "name": "ad",
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
            },
            {
                "name": "adosc",
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "adx",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "adxr",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "ao",
                "primary_columns": ["high", "low"],
                "output_columns": 1,
            },
            {
                "name": "apo",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "apo",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "apo",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "apo",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "aroon",
                "primary_columns": ["high", "low"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "aroonosc",
                "primary_columns": ["high", "low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "atr",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "bop",
                "primary_columns": ["open", "high", "low", "close"],
                "output_columns": 1,
            },
            {
                "name": "cci",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "cmo",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "cmo",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "cmo",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "cmo",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "cvi",
                "primary_columns": ["high", "low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "di",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "dm",
                "primary_columns": ["high", "low"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "dpo",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "dpo",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "dpo",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "dpo",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "dx",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "emv",
                "primary_columns": ["high", "low", "volume"],
                "output_columns": 1,
            },
            {
                "name": "fisher",
                "primary_columns": ["high", "low"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "fosc",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "fosc",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "fosc",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "fosc",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "fosc",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "kvo",
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "linregintercept",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregintercept",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregintercept",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregintercept",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregslope",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregslope",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregslope",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "linregslope",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "macd",
                "primary_columns": ["open"],
                "output_columns": 3,
                "period_list": (2, 5, 9),
            },
            {
                "name": "macd",
                "primary_columns": ["high"],
                "output_columns": 3,
                "period_list": (2, 5, 9),
            },
            {
                "name": "macd",
                "primary_columns": ["low"],
                "output_columns": 3,
                "period_list": (2, 5, 9),
            },
            {
                "name": "macd",
                "primary_columns": ["close"],
                "output_columns": 3,
                "period_list": (2, 5, 9),
            },
            {
                "name": "marketfi",
                "primary_columns": ["high", "low", "volume"],
                "output_columns": 1,
            },
            {
                "name": "mass",
                "primary_columns": ["high", "low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "mfi",
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "mom",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "mom",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "mom",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "mom",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "msw",
                "primary_columns": ["open"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "msw",
                "primary_columns": ["high"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "msw",
                "primary_columns": ["low"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "msw",
                "primary_columns": ["close"],
                "output_columns": 2,
                "period_list": (5,),
            },
            {
                "name": "natr",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "nvi",
                "primary_columns": ["close", "volume"],
                "output_columns": 1,
            },
            {
                "name": "obv",
                "primary_columns": ["close", "volume"],
                "output_columns": 1,
            },
            {
                "name": "ppo",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "ppo",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "ppo",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "ppo",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "pvi",
                "primary_columns": ["close", "volume"],
                "output_columns": 1,
            },
            {
                "name": "qstick",
                "primary_columns": ["open", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "roc",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "roc",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "roc",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "roc",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rocr",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rocr",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rocr",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rocr",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rsi",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rsi",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rsi",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "rsi",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "stoch",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 2,
                "period_list": (5, 3, 3),
            },
            {
                "name": "stochrsi",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "stochrsi",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "stochrsi",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "stochrsi",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "tr",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
            },
            {
                "name": "trix",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "trix",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "trix",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "trix",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "ultosc",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (2, 3, 5),
            },
            {
                "name": "volatility",
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "volatility",
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "volatility",
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "volatility",
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": (5,),
            },
            {
                "name": "vosc",
                "primary_columns": ["volume"],
                "output_columns": 1,
                "period_list": (2, 5),
            },
            {
                "name": "wad",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
            },
            {
                "name": "willr",
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": (5,),
            },
        ]

    def get_permutations(self):
        with open("indicator_combinations.txt", "a") as fd:
            for r in range(min(4, len(self.indicator_dict) + 1)):
                for element in combinations(self.indicator_dict, r):
                    fd.write(str(list(element)) + "\n")
        return

    def get_params(self, param_name):
        return self.indicator_dict[param_name]


if __name__ == "__main__":
    ind_dic = indicator_dict()
    ind_dic.get_permutations()
    # with open("indicator_combinations.txt", "r") as fd:
    #     lines = fd.readlines()
    #     for line in lines:
    #         indicator_lst = ast.literal_eval(line)
    #         for indicator in indicator_lst:
    #             print(ind_dic.get_params(indicator))
