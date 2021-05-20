"""Module containing the dictionary of indicators and their parameters."""
from itertools import combinations


class IndicatorDict:

    """Class containing the dictionary of indicators and their parameters."""

    def __init__(self):
        """Initialize the dictionary."""
        self.indicator_dict = {
            "ad": {
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
            },
            "adosc": {
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "adx": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "adxr": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "ao": {
                "primary_columns": ["high", "low"],
                "output_columns": 1,
            },
            "apoOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "apoHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "apoLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "apoCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "aroon": {
                "primary_columns": ["high", "low"],
                "output_columns": 2,
                "period_list": [5],
            },
            "aroonosc": {
                "primary_columns": ["high", "low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "atr": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "bop": {
                "primary_columns": ["open", "high", "low", "close"],
                "output_columns": 1,
            },
            "cci": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "cmoOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "cmoHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "cmoLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "cmoCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "cvi": {
                "primary_columns": ["high", "low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "di": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 2,
                "period_list": [5],
            },
            "dm": {
                "primary_columns": ["high", "low"],
                "output_columns": 2,
                "period_list": [5],
            },
            "dpoOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "dpoHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "dpoLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "dpoCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "dx": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "emv": {
                "primary_columns": ["high", "low", "volume"],
                "output_columns": 1,
            },
            "fisher": {
                "primary_columns": ["high", "low"],
                "output_columns": 2,
                "period_list": [5],
            },
            "foscOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "foscHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "foscLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "foscCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "fosc": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "kvo": {
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "linreginterceptOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linreginterceptHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linreginterceptLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linreginterceptCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linregslopeOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linregslopeHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linregslopeLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "linregslopeCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "macdOPEN": {
                "primary_columns": ["open"],
                "output_columns": 3,
                "period_list": [2, 5, 9],
            },
            "macdHIGH": {
                "primary_columns": ["high"],
                "output_columns": 3,
                "period_list": [2, 5, 9],
            },
            "macdLOW": {
                "primary_columns": ["low"],
                "output_columns": 3,
                "period_list": [2, 5, 9],
            },
            "macdCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 3,
                "period_list": [2, 5, 9],
            },
            "marketfi": {
                "primary_columns": ["high", "low", "volume"],
                "output_columns": 1,
            },
            "mass": {
                "primary_columns": ["high", "low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "mfi": {
                "primary_columns": ["high", "low", "close", "volume"],
                "output_columns": 1,
                "period_list": [5],
            },
            "momOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "momHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "momLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "momCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "mswOPEN": {
                "primary_columns": ["open"],
                "output_columns": 2,
                "period_list": [5],
            },
            "mswHIGH": {
                "primary_columns": ["high"],
                "output_columns": 2,
                "period_list": [5],
            },
            "mswLOW": {
                "primary_columns": ["low"],
                "output_columns": 2,
                "period_list": [5],
            },
            "mswCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 2,
                "period_list": [5],
            },
            "natr": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "nvi": {
                "primary_columns": ["close", "volume"],
                "output_columns": 1,
            },
            "obv": {
                "primary_columns": ["close", "volume"],
                "output_columns": 1,
            },
            "ppoOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "ppoHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "ppoLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "ppoCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "pvi": {
                "primary_columns": ["close", "volume"],
                "output_columns": 1,
            },
            "qstick": {
                "primary_columns": ["open", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocrOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocrHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocrLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rocrCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rsiOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rsiHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rsiLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "rsiCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "stoch": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 2,
                "period_list": [5, 3, 3],
            },
            "stochrsiOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "stochrsiHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "stochrsiLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "stochrsiCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "tr": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
            },
            "trixOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "trixHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "trixLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "trixCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "ultosc": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [2, 3, 5],
            },
            "volatilityOPEN": {
                "primary_columns": ["open"],
                "output_columns": 1,
                "period_list": [5],
            },
            "volatilityHIGH": {
                "primary_columns": ["high"],
                "output_columns": 1,
                "period_list": [5],
            },
            "volatilityLOW": {
                "primary_columns": ["low"],
                "output_columns": 1,
                "period_list": [5],
            },
            "volatilityCLOSE": {
                "primary_columns": ["close"],
                "output_columns": 1,
                "period_list": [5],
            },
            "vosc": {
                "primary_columns": ["volume"],
                "output_columns": 1,
                "period_list": [2, 5],
            },
            "wad": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
            },
            "willr": {
                "primary_columns": ["high", "low", "close"],
                "output_columns": 1,
                "period_list": [5],
            },
        }

    def get_permutations(self):
        """Get all permutations of combinations."""
        with open("indicator_combinations.txt", "a") as write_file:
            for each in range(min(4, len(self.indicator_dict) + 1)):
                for element in combinations(self.indicator_dict, each):
                    write_file.write(str(list(element)) + "\n")

    def get_params(self, param_name):
        """Get the parameters of a single indicator."""
        return self.indicator_dict[param_name]


if __name__ == "__main__":
    ind_dic = IndicatorDict()
    ind_dic.get_permutations()
    # with open("indicator_combinations.txt", "r") as fd:
    #     lines = fd.readlines()
    #     for line in lines:
    #         indicator_lst = ast.literal_eval(line)
    #         for indicator in indicator_lst:
    #             print(ind_dic.get_params(indicator))
