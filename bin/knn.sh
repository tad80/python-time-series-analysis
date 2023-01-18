#!/bin/bash
#pyenv shell 3.9.14
export GOOGLE_APPLICATION_CREDENTIALS="/home/tadashi.a.nakamura/.keio-sdm-masters-research-9d35c386c90e.json"
pipenv run python knn_analyzer.py --bq_conf $1 --location $2 --key $3 --target_month $4 --normalize $5
