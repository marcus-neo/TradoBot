import os


class Parameters:
    ROOT_DIRECTORY = "/home/marcus/PersonalGit/TradoBot"
    TICKER_LIST_DIRECTORY = os.path.join(
        ROOT_DIRECTORY, "StockTickers/TickerNames.csv"
    )
    IMAGE_OUTPUT_DIRECTORY = os.path.join(ROOT_DIRECTORY, "output/images")
    LABEL_OUTPUT_DIRECTORY = os.path.join(ROOT_DIRECTORY, "output/labels")
    TEST_OUTPUT_DIRECTORY = os.path.join(ROOT_DIRECTORY, "output/test")
    SAMPLES = 4000
    SAMPLE_WINDOW_SIZE = 730
    LOOKBACK = 50
    LOOKAHEAD = 5
    IMAGE_HEIGHT = 320
    IMAGE_WIDTH = 320
    SUCCESSFUL_TRADE_PERC = 5
