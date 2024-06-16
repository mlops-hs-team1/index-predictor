import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from typing import Optional
import argparse


class DataCollector:
    def __init__(
        self,
        filename: str,
        days: int = 30,
        ticker: str = "^GSPC",
        num_rows: Optional[int] = None,
    ):
        self.data_folder = "data/raw"
        self.filename = filename
        self.days = days
        self.ticker = ticker
        self.num_rows = num_rows

    def get_data(self) -> pd.DataFrame:
        full_data = pd.DataFrame()

        end = datetime.now()
        start = end - pd.Timedelta(days=self.days)

        while start < end:
            next_time = start + pd.Timedelta(days=7)
            if next_time > end:
                next_time = end

            data = yf.download(self.ticker, start=start, end=next_time, interval="1m")

            if not data.empty:
                full_data = pd.concat([full_data, data])

            start = next_time

        if self.num_rows:
            full_data = full_data.tail(self.num_rows)

        return full_data

    def store_data(self, data: pd.DataFrame):
        os.makedirs(self.data_folder, exist_ok=True)
        data.to_csv(f"{self.data_folder}/{self.filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Mode to run the script, either 'all', 'train-val-test' or 'inference'",
    )
    parser.add_argument(
        "--datapoints",
        type=int,
        default=31,
        help="Number of data points to collect",
    )

    args = parser.parse_args()

    if args.mode not in ["all", "train-val-test", "inference"]:
        raise ValueError("Invalid mode")
    if args.mode in ["all", "train-val-test"]:
        collector = DataCollector(
            filename=f"data.csv",
            days=30,
            ticker="^GSPC",
            num_rows=None,
        )
        data = collector.get_data()
        collector.store_data(data)
    if args.mode in ["all", "inference"]:
        collector = DataCollector(
            filename="data_inference.csv",
            days=3,
            ticker="^GSPC",
            num_rows=args.datapoints,
        )
        data = collector.get_data()
        collector.store_data(data)
