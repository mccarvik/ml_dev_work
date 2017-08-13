import sys
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime, pdb, time
import numpy as np
from utils.helper_funcs import timeme
from utils.data_utils import *
from utils.db_utils import DBHelper
from scripts.ml_algorithms import *


def run(inputs):
    # Temp to make testing quicker
    t0 = time.time()
    with DBHelper() as db:
        db.connect()
        df = db.select('morningstar', where = 'date in ("2006", "2007", "2008", \
                        "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016")')
    # Getting Dataframe
    # df = getKeyStatsDataFrame(table='morningstar', date='')
    t1 = time.time()
    # app.logger.info("Done Retrieving data, took {0} seconds".format(t1-t0))
    print("Done Retrieving data, took {0} seconds".format(t1-t0))
    
    # Set final inputs here, need other ones previous to this for pruning
    # inputs = ['trailingPE', 'returnOnEquity']
    
    df = removeUnnecessaryColumns(df)
    df = addTarget(df, '5yrReturn')
    df = cleanData(df)
    df = selectInputs(df, inputs)
    df = df.reset_index().drop('index', 1)
    print("There are {0} samples".format(len(df)))
    
    # timeme(logisticRegression)(df, tuple(inputs), C=1000, penalty='l1')
    # timeme(support_vector_machines)(df, tuple(inputs), C=1)
    # timeme(nonlinear_svm)(df, tuple(inputs), C=1)
    # timeme(decision_tree)(df, tuple(inputs), md=4)
    # timeme(random_forest)(df, tuple(inputs), estimators=3)
    # timeme(k_nearest_neighbors)(df, tuple(inputs), k=8)
    # timeme(sbs_run)(df, tuple(inputs))
    # timeme(random_forest_feature_importance)(df, tuple(inputs))
    # timeme(principal_component_analysis)(df, tuple(inputs))
    # timeme(pca_scikit)(df, tuple(inputs))
    # timeme(linear_discriminant_analysis)(df, tuple(inputs))
    timeme(lda_scikit)(df, tuple(inputs))
    

def selectInputs(df, inputs):
    columns = inputs + ['target'] + ['target_proxy']
    df = df[columns]
    return df

def addTarget(df, tgt):
    num_of_breaks = 5
    df['target_proxy'] = df[tgt]
    df = df.dropna(subset = ['target_proxy'])
    df = df[df['target_proxy'] != 0]
    
    break_arr = np.linspace(0, 100, num_of_breaks+1)[1:-1]
    breaks = np.percentile(df['target_proxy'], break_arr)
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
    df = df[abs(df['trailingPE']) < 30]
    # df = df[abs(df['priceToBook']) < 10]
    df = df[df['trailingPE'] > 0]
    df = df[df['divYield'] > 0]
    df = df[df['divYield'] < 8]
    # df = df[df['debtToEquity'] < 10]
    # df = df[df['returnOnEquity'] > 0]
    df = df[df['returnOnEquity'] < 50]
    # df = df[df['currentRatio'] < 10]
    

    # only look at the top and bottom percentile ranges
    df = df[(df['target'] == 0) | (df['target'] == 4)]
    # pdb.set_trace()
    return df

if __name__ == "__main__":
    run(['trailingPE', 'priceToBook', 'priceToSales', 'divYield', 'debtToEquity',
        'returnOnEquity', 'netIncomeMargin', 'freeCashFlowPerShare', 'currentRatio',
        'quickRatio','financialLeverage','capExToSales', 'priceToCashFlow'])