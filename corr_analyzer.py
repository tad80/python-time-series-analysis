import argparse
import pandas as pd
import numpy as np
import os
import seaborn as sns
from google.cloud import storage
from configparser import ConfigParser
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from libs.bigquery_client import BigQueryClient
from libs.logger import Logger

class CorrAnalyzer:

    def __init__(self, bq_conf, location, key):
        """
        Init constructor
        """
        self.logger = Logger("./config/logger.ini", self.__class__.__name__)
        self.bq = BigQueryClient(bq_conf)
        self.location = location
        self.key = key


    def __sentiment(self):
        sql = """
SELECT
  concat(`year`, '-', format('%02d', `month`), '-01') dt,
  `tweets`,
  `sentiment`
FROM (
  SELECT
    extract(YEAR from `created_at`) year,
    extract(MONTH from `created_at`) month,
    count(1) tweets,
    avg(score) sentiment
  FROM
    `keio-sdm-masters-research.tweets.tweets_sentiment_v1`
  where
    location = '{}' and
    created_at >= '2012-11-01' and
    created_at < '2014-12-01'
  group by
    extract(YEAR from `created_at`),
    extract(MONTH from `created_at`)
)
order by
  `year`,
  `month`
""".format(self.location)
        self.logger.log.info("Executing BQ query: %s" % sql)
        return self.bq.client.query(sql).to_dataframe()


    def __revenue(self):
        sql = """
SELECT
  *
FROM
  `keio-sdm-masters-research.tweets.internal_revenue`
WHERE
  location = '{}'
ORDER BY
  dt
""".format(self.location)
        self.logger.log.info("Executing BQ query: %s" % sql)
        return self.bq.client.query(sql).to_dataframe()


    def main(self):
        """
        Main method
        """
        sentiment = self.__sentiment()
        print(sentiment)
        revenue = self.__revenue()
        print(revenue)
        df = pd.DataFrame()
        df['month'] = pd.to_datetime(sentiment['dt'], format = "%Y-%m-%d").dt.date
        df['sentiment'] = sentiment['sentiment']
        df['revenue'] = revenue['revenue']
        print(df.head())
        fig, ax = plt.subplots(figsize = (10, 6))
        ax.plot(df['month'],
            df['revenue'],
            color="red", 
            marker="o")
        ax.set_xlabel("month", fontsize = 14)
        ax.set_ylabel("revenue",
            color="red",
            fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(df['month'],
            df['sentiment'],
            color="blue",
            marker="o")
        ax2.set_ylabel("sentiment",
            color="blue",
            fontsize=14)
        plt.title('[%s] Twitter sentiment vs internal revenue' % self.location, weight='bold', fontsize = 15)
        filename = "%s_xcorr.png" % (self.location)
        plt.savefig("resources/%s" % filename)
        with open("resources/%s" % filename, mode="rb") as f:
            gcs = storage.Client()
            bucket = gcs.get_bucket("cs.gcp.tdsnkm.com")
            blob = bucket.blob("xcorr/%s" % filename)
            blob.upload_from_string(
                f.read(),
                content_type="image/png"
            )
        os.remove("resources/%s" % filename)

        seasonal = pd.DataFrame()
        seasonal[self.key] = df[self.key]
        seasonal.index = pd.to_datetime(df['month'], format = "%Y-%m-%d").dt.date
        print(seasonal.head())
        figure(figsize = (10, 6))
        result = seasonal_decompose(seasonal, model='additive', period=12)
        result.plot()
        filename = "%s_%s_tsa_additive.png" % (self.location, self.key)
        plt.savefig("resources/%s" % filename)
        with open("resources/%s" % filename, mode="rb") as f:
            gcs = storage.Client()
            bucket = gcs.get_bucket("cs.gcp.tdsnkm.com")
            blob = bucket.blob("xcorr/%s" % filename)
            blob.upload_from_string(
                f.read(),
                content_type="image/png"
            )
        os.remove("resources/%s" % filename)

        result = seasonal_decompose(seasonal, model='multiplicative', period=12)
        result.plot()
        filename = "%s_%s_tsa_multiplicative.png" % (self.location, self.key)
        plt.savefig("resources/%s" % filename)
        with open("resources/%s" % filename, mode="rb") as f:
            gcs = storage.Client()
            bucket = gcs.get_bucket("cs.gcp.tdsnkm.com")
            blob = bucket.blob("xcorr/%s" % filename)
            blob.upload_from_string(
                f.read(),
                content_type="image/png"
            )
        os.remove("resources/%s" % filename)

        result = STL(seasonal, period=12, robust=True).fit()
        print(result.resid)
        result.plot()
        filename = "%s_%s_tsa_stl.png" % (self.location, self.key)
        plt.savefig("resources/%s" % filename)
        with open("resources/%s" % filename, mode="rb") as f:
            gcs = storage.Client()
            bucket = gcs.get_bucket("cs.gcp.tdsnkm.com")
            blob = bucket.blob("xcorr/%s" % filename)
            blob.upload_from_string(
                f.read(),
                content_type="image/png"
            )
        os.remove("resources/%s" % filename)


if __name__ == "__main__":
    try:
        ARGPARSER = argparse.ArgumentParser(description="Analyze time series described in the file")
        ARGPARSER.add_argument("--bq_conf", required=True, help="BigQuery configuration file")
        ARGPARSER.add_argument("--location", required=True, help="Location string")
        ARGPARSER.add_argument("--key", required=True, help="sentiment or revenue")
        ARGS = ARGPARSER.parse_args()
        ANALYZER = CorrAnalyzer(ARGS.bq_conf, ARGS.location, ARGS.key)
        ANALYZER.main()
    except Exception as ex:
        raise
