import argparse
import pandas as pd
from google.cloud import storage
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from configparser import ConfigParser
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
from libs.logger import Logger

class ProphetAnalyzer:

    def __init__(self, file, field):
        """
        Init constructor
        """
        self.logger = Logger("./config/logger.ini", self.__class__.__name__)
        self.file = file
        self.field = field


    def main(self):
        """
        Main method
        """
        ts = pd.read_table(self.file)
        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(ts['month'])
        df['y'] = ts[self.field]
        df.head
        #prophet = Prophet(changepoint_prior_scale=0.001)
        prophet = Prophet()
        prophet.fit(df)
        future = prophet.make_future_dataframe(periods=12 * 2, freq='M')
        forecast = prophet.predict(future)
        fig = prophet.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), prophet, forecast)
        fig.savefig("leyte_%s.png" % self.field)
        with open("leyte_%s.png" % self.field, mode="rb") as f:
            gcs = storage.Client()
            bucket = gcs.get_bucket("cs.gcp.tdsnkm.com")
            blob = bucket.blob("leyte_%s.png" % self.field)
            blob.upload_from_string(
                f.read(),
                content_type="image/png"
            )


if __name__ == "__main__":
    try:
        ARGPARSER = argparse.ArgumentParser(description="Analyze time series described in the file")
        ARGPARSER.add_argument("--file", required=True, help="Input time series file")
        ARGPARSER.add_argument("--field", required=True, help="Analysis target field")
        ARGS = ARGPARSER.parse_args()
        ANALYZER = ProphetAnalyzer(ARGS.file, ARGS.field)
        ANALYZER.main()
    except Exception as ex:
        raise

