#!/bin/bash
#pyenv shell 3.9.14
export GOOGLE_APPLICATION_CREDENTIALS="/home/tadashi.a.nakamura/.keio-sdm-masters-research-9d35c386c90e.json"
pipenv run python prophet_analyzer.py --file $1 --field $2
