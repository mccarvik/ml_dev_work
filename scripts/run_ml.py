import sys
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime, pdb, time
import numpy as np
import pandas as pd
from utils.helper_funcs import timeme
from utils.data_utils import *
from utils.db_utils import DBHelper
from scripts.ml_algorithms import *


def run(inputs):
    # Temp to make testing quicker
    t0 = time.time()
    with DBHelper() as db:
        db.connect()
        df = db.select('morningstar', where = 'date in ("2010", "2015")')
    # Getting Dataframe
    # df = getKeyStatsDataFrame(table='morningstar', date='')
    t1 = time.time()
    # app.logger.info("Done Retrieving data, took {0} seconds".format(t1-t0))
    print("Done Retrieving data, took {0} seconds".format(t1-t0))
    
    # Set final inputs here, need other ones previous to this for pruning
    inputs = ['currentRatio', 'debtToEquity']
    
    df = removeUnnecessaryColumns(df)
    df = timeme(addTarget)(df)
    df = cleanData(df)
    df = selectInputs(df, inputs)
    df = df.reset_index().drop('index', 1)
    print("There are {0} samples".format(len(df)))
    
    # timeme(logisticRegression)(df, tuple(inputs))
    timeme(support_vector_machines)(df, tuple(inputs))
    # timeme(run_perceptron_multi)(df, tuple(inputs))

def selectInputs(df, inputs):
    columns = inputs + ['target'] + ['target_proxy']
    df = df[columns]
    return df

def addTarget(df):
    yr_avg_ret = 5
    target = []
    for ind, row in df.iterrows():
        try:
            t = df[(df['ticker'] == row['ticker']) & (df['date'] == str(int(row['date']) + yr_avg_ret))]['5yrReturn'].iloc[0]
            target.append(t)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            target.append(np.nan)
    df['target_proxy'] = target
    df = df.dropna(subset = ['target_proxy'])
    df = df[df['target_proxy'] != 0]
    breaks = np.percentile(df['target_proxy'], [25, 50, 75])
    # breaks = np.percentile(df['target_proxy'], [50])
    df['target'] = df.apply(lambda x: targetToCatMulti(x['target_proxy'], breaks), axis=1)
    return df

def targetToCat(x, median):
    if (x > median):
        return 1
    else:
        return -1

def targetToCatMulti(x, breaks):
    cat = 0
    for b in breaks:
        if x < b:
            return cat
        cat += 1
    return cat

def removeUnnecessaryColumns(df):
    df = df[RATIOS + KEY_STATS + OTHER +
            GROWTH + MARGINS + RETURNS +
            PER_SHARE + INDEX]
    return df

def cleanData(df):
    # To filter out errant data
    df = df[df['trailingPE'] != 0]
    df = df[df['priceToBook'] > 0]
    df = df[df['priceToSales'] != 0]
    df = df[df['divYield'] >= 0]
    
    # Temp for training purposes
    df = df[abs(df['trailingPE']) < 100]
    # df = df[abs(df['priceToBook']) < 10]
    df = df[df['trailingPE'] > 0]
    # df = df[df['divYield'] > 0]
    # df = df[df['divYield'] < 10]
    df = df[df['debtToEquity'] < 10]
    # df = df[df['returnOnEquity'] > 0]
    # df = df[df['returnOnEquity'] < 50]
    # df = df[df['currentRatio'] < 10]
    # df = df[df[''] > 0]
    

    # only look at the top 25 and bottom 25%
    df = df[(df['target'] == 0) | (df['target'] == 3)]
    # pdb.set_trace()
    return df

if __name__ == "__main__":
    run(['trailingPE', 'priceToBook', 'priceToSales', 'divYield', 'debtToEquity',
        'returnOnEquity', 'netIncomeMargin', 'freeCashFlowPerShare', 'currentRatio',
        'quickRatio','financialLeverage','capExToSales', 'priceToCashFlow'])