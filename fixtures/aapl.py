import os
from pathlib import Path
import csv
import pytest

fixtures_directory = Path(__file__).resolve().parent


@pytest.fixture
def aapl_test():
    aapl_directory = os.path.join(fixtures_directory, "aapl")
    ticker_directory = os.path.join(
        fixtures_directory, "aapl", "TickerNames.csv"
    )
    predictions_filename = "output.csv"

    with open(
        os.path.join(aapl_directory, predictions_filename)
    ) as predictions_file:
        predictions = list(csv.reader(predictions_file))
    return (ticker_directory, predictions)
