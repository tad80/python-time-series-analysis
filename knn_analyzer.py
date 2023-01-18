import argparse
import pandas as pd
import numpy as np
import os
import sys
from google.cloud import storage
from configparser import ConfigParser
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from libs.bigquery_client import BigQueryClient
from libs.logger import Logger

class KNNAnalyzer:

    def __init__(self, bq_conf, key, target_month, standardize):
        """
        Init constructor
        """
        self.logger = Logger("./config/logger.ini", self.__class__.__name__)
        self.bq = BigQueryClient(bq_conf)
        self.key = key
        self.target_month = target_month
        self.standardize = standardize


    def __select(self):
        sql = """
SELECT
  `location`,
  concat(`year`, '-', format('%02d', `month`), '-', format('%02d', `day`)) dt,
  `tweets`,
  `sentiment`
FROM (
  SELECT
    `location`,
    extract(YEAR from `created_at`) year,
    extract(MONTH from `created_at`) month,
    extract(DAY from `created_at`) day,
    count(1) tweets,
    avg(score) sentiment
  FROM
    `keio-sdm-masters-research.tweets.tweets_sentiment_v1`
  where
    -- location not in ('manila', 'laguna', 'batangas', 'cavite')
    location = 'biliran'
  group by
    `location`,
    extract(YEAR from `created_at`),
    extract(MONTH from `created_at`),
    extract(DAY from `created_at`)
)
order by
  `location`,
  `year`,
  `month`,
  `day`
"""
        self.logger.log.info("Executing BQ query: %s" % sql)
        return self.bq.client.query(sql).to_dataframe()


    def main(self):
        """
        Main method
        """
        df = self.__select()
        self.logger.log.info("displaying whole dataframe")
        print(df)
        tweets = df[['tweets']]
        sentiments = df[['sentiment']]
        dtype = 'raw'
        if self.standardize == 'standardize':
            dtype = 'standardized'
            tw_scaler = StandardScaler().fit(tweets)
            tw_scaled = tw_scaler.transform(tweets)
            #np.set_printoptions(threshold=sys.maxsize)
            print(tw_scaled)
            print(np.amax(tw_scaled))
            st_scaler = StandardScaler().fit(sentiments)
            st_scaled = st_scaler.transform(sentiments)
            print(st_scaled)
            print(np.amax(st_scaled))
            df['tweets'] = tw_scaled
            df['sentiment'] = st_scaled

        #locations = ['aklan', 'albay', 'bacolod', 'bohol', 'camarines_sur', 'capiz', 'cebu', 'iloilo', 'leyte', 'masbate', 'negros_occidental', 'palawan', 'samar', 'sorsogon']
        locations = ['biliran']
        for location in locations:
            validation_target = df.loc[(df['location'] == location) & (df['dt'].str.startswith(self.target_month))]
            month = datetime.strptime(self.target_month, '%Y-%m').strftime('%m')
            training_target = df.loc[(df['location'] == location) & (~df['dt'].str.startswith(self.target_month)) & (df['dt'].str.contains("^20[0-9]{2}-%s-[0-9]{2}$" % month))]
            self.logger.log.info("displaying training target dataframe for %s" % location)
            print(training_target)
            csv = "%s_%s_%s_%s_%s_knn.csv" % (location, self.target_month, self.key, dtype, 'train')
            training_target.to_csv('resources/%s' % csv)
            self.logger.log.info("displaying validation target dataframe for %s" % location)
            print(validation_target)

            ## 窓幅
            width = 30
            ## K近傍法のk
            nk = 1
            ## 窓幅を使ってベクトルの集合を作成
            print(training_target[self.key])
            ## k近傍法でクラスタリング
            train = np.array(training_target[self.key]).reshape(-1, 1)
            test = np.array(validation_target[self.key]).reshape(-1, 1)
            neigh = NearestNeighbors(n_neighbors=nk)
            neigh.fit(train)
            ## 距離を計算
            distance = neigh.kneighbors(test)[0]
            print(distance)
            validation_target = validation_target.assign(distance=distance)
            csv = "%s_%s_%s_%s_%s_knn.csv" % (location, self.target_month, self.key, dtype, 'distance')
            validation_target.to_csv('resources/%s' % csv)
            ## グラフ作成
            fig = plt.figure(figsize=(10.0, 8.0))
            #plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            ## 訓練データ
            plt.subplot(221)
            plt.plot(training_target[self.key], label='Training')
            plt.xlabel("Amplitude", fontsize=12)
            plt.ylabel("Sample", fontsize=12)
            plt.grid()
            leg = plt.legend(loc=1, fontsize=15)
            leg.get_frame().set_alpha(1)
            ## 異常度
            plt.subplot(222)
            plt.plot(distance, label='d')
            plt.xlabel("Amplitude", fontsize=12)
            plt.ylabel("Sample", fontsize=12)
            plt.grid()
            leg = plt.legend(loc=1, fontsize=15)
            leg.get_frame().set_alpha(1)
            ## 検証用データ
            plt.subplot(223)
            plt.plot(validation_target[self.key], label='Test')
            plt.xlabel("Amplitude", fontsize=12)
            plt.ylabel("Sample", fontsize=12)
            plt.grid()
            leg = plt.legend(loc=1, fontsize=15)
            leg.get_frame().set_alpha(1)
            plt.subplot(224)
            #plt.savefig('C:/github/sample/python/scikit/kmeans/sample1.png')
            filename = "%s_%s_%s_%s_knn.png" % (location, self.target_month, self.key, dtype)
            plt.savefig(filename)
            with open(filename, mode="rb") as f:
                gcs = storage.Client()
                bucket = gcs.get_bucket("cs.gcp.tdsnkm.com")
                blob = bucket.blob("knn_sentiment/%s" % filename)
                blob.upload_from_string(
                    f.read(),
                    content_type="image/png"
                )
            os.remove(filename)


if __name__ == "__main__":
    try:
        ARGPARSER = argparse.ArgumentParser(description="Analyze time series described in the file")
        ARGPARSER.add_argument("--bq_conf", required=True, help="BigQuery configuration file")
        ARGPARSER.add_argument("--key", required=True, help="tweets or sentiment")
        ARGPARSER.add_argument("--target_month", required=True, help="Target month")
        ARGPARSER.add_argument("--standardize", required=True, help="standardize or raw, whether you want to standardize the distance")
        ARGS = ARGPARSER.parse_args()
        ANALYZER = KNNAnalyzer(ARGS.bq_conf, ARGS.key, ARGS.target_month, ARGS.standardize)
        ANALYZER.main()
    except Exception as ex:
        raise

