import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])

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
        last_timestamp: Optional[str] = None,
        outputpath: str = "/opt/ml/processing/output/",
    ):
        self.data_folder = "data/raw"
        self.filename = filename
        self.days = days
        self.ticker = ticker
        self.num_rows = num_rows
        self.outputpath = outputpath
        self.last_timestamp = pd.to_datetime(last_timestamp) if last_timestamp else None

    def get_data(self) -> pd.DataFrame:
        full_data = pd.DataFrame()

        end = datetime.now() if not self.last_timestamp else self.last_timestamp
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
        folder = os.path.join(self.outputpath, self.data_folder)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, self.filename)
        data.to_csv(path)

        print(f"Data stored at {path}")


def parse_args():
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
    parser.add_argument(
        "--last_timestamp",
        type=str,
        default=None,
        help="Last timestamp for data collection",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default="/opt/ml/processing/output/",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode not in ["all", "train-val-test", "inference"]:
        raise ValueError("Invalid mode")
    if args.mode in ["all", "train-val-test"]:
        collector = DataCollector(
            filename=f"data.csv",
            days=30,
            ticker="^GSPC",
            num_rows=None,
            last_timestamp=args.last_timestamp,
            outputpath=args.outputpath,
        )
        data = collector.get_data()
        collector.store_data(data)
    if args.mode in ["all", "inference"]:
        collector = DataCollector(
            filename=f"data-inference.csv",
            days=3,
            ticker="^GSPC",
            num_rows=args.datapoints,
            last_timestamp=args.last_timestamp,
            outputpath=args.outputpath,
        )
        data = collector.get_data()
        collector.store_data(data)

    print("Data collection completed")
