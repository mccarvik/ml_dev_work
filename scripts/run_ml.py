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
from scripts.model_evaluation import *
from scripts.feature_selection import *
from scripts.ensemble_methods import *
from scripts.continuous_variables import *


def run(inputs):
    # Temp to make testing quicker
    t0 = time.time()
    tickers = pd.read_csv('/home/ubuntu/workspace/ml_dev_work/utils/snp500_ticks.csv', header=None)
    # tickers = pd.read_csv('/home/ubuntu/workspace/ml_dev_work/utils/dow_ticks.csv', header=None)
    with DBHelper() as db:
        db.connect()
        # df = db.select('morningstar', where = 'date in ("2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016")')
        lis = ''
        for t in list(tickers[0]):
            lis += t + ", "
        df = db.select('morningstar', where = 'ticker in (' + lis[:-2] + ')')
        
    # Getting Dataframe
    t1 = time.time()
    print("Done Retrieving data, took {0} seconds".format(t1-t0))
    
    # grab the more recent data for testing later
    test_df, df = separateTrainTest(df)
    
    df = removeUnnecessaryColumns(df)
    df = cleanData(df)
    df = addTarget(df, '3yrFwdReturn')
    df = df.set_index(['ticker', 'date'])
    df = selectInputs(df, inputs)
    print("There are {0} samples".format(len(df)))
    pdb.set_trace()
    
    # TODO: Do feature extraction from here
    # Sequential Backward Selection - feature selection to see which are the most telling variable
    timeme(sbs_run)(df, tuple(inputs))
    # timeme(logisticRegression)(df, tuple(inputs), C=1000, penalty='l1')
    # timeme(support_vector_machines)(df, tuple(inputs), C=1)
    # timeme(nonlinear_svm)(df, tuple(inputs), C=1)
    # timeme(decision_tree)(df, tuple(inputs), md=4)
    # timeme(random_forest)(df, tuple(inputs), estimators=3)
    # timeme(k_nearest_neighbors)(df, tuple(inputs), k=8)
    # timeme(sbs_run)(df, tuple(inputs), est=DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0))
    # timeme(random_forest_feature_importance)(df, tuple(inputs))
    # timeme(principal_component_analysis)(df, tuple(inputs))
    # timeme(pca_scikit)(df, tuple(inputs))
    # timeme(linear_discriminant_analysis)(df, tuple(inputs))
    # timeme(lda_scikit)(df, tuple(inputs))
    # timeme(kfold_cross_validation)(df, tuple(inputs))
    # timeme(learning_curves)(df, tuple(inputs))
    # timeme(validation_curves)(df, tuple(inputs))
    # timeme(grid_search_analysis)(df, tuple(inputs))
    # timeme(precision_vs_recall)(df, tuple(inputs))
    # timeme(majority_vote)(df, tuple(inputs))
    # timeme(bagging)(df, tuple(inputs))
    # timeme(adaboost)(df, tuple(inputs))
    # timeme(heat_map)(df, tuple(inputs))
    # timeme(linear_regressor)(df, tuple(inputs))
    # timeme(linear_regression_sklearn)(df, tuple(inputs))
    # timeme(ransac)(df, tuple(inputs))
    # timeme(polynomial_regression)(df, tuple(inputs))
    # timeme(nonlinear)(df, tuple(inputs))
    # timeme(random_forest_regression)(df, tuple(inputs))


def separateTrainTest(df):
    pdb.set_trace()
    test_df = df[df.date == '2017']
    df = df[df.date != '2017']
    return test_df, df


def selectInputs(df, inputs):
    columns = inputs + ['target'] + ['target_proxy']
    df = df[columns]
    return df


def addTarget(df, tgt):
    num_of_breaks = 2
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
    df = df[RATIOS + KEY_STATS + OTHER + GROWTH + MARGINS + RETURNS + PER_SHARE + INDEX]
    return df


def cleanData(df):
    # To filter out errant data
    df = df[df['trailingPE'] != 0]
    df = df[df['priceToBook'] > 0]
    df = df[df['priceToSales'] != 0]
    df = df[df['3yrFwdReturn'] != 0]
    
    # Temp for training purposes
    # df = df[abs(df['trailingPE']) < 30]
    # df = df[abs(df['priceToBook']) < 10]
    # df = df[df['trailingPE'] > 0]
    # df = df[df['divYield'] > 0]
    # df = df[df['divYield'] < 8]
    # df = df[df['debtToEquity'] < 10]
    # df = df[df['returnOnEquity'] > 0]
    # df = df[df['returnOnEquity'] < 50]
    # df = df[df['currentRatio'] < 10]
    

    # only look at the top and bottom percentile ranges
    # df = df[(df['target'] == 0) | (df['target'] == 4)]
    return df


if __name__ == "__main__":
    run(['trailingPE', 'priceToBook', 'priceToSales', 'divYield', 'debtToEquity', 'returnOnEquity', 'netIncomeMargin', 
        'freeCashFlowPerShare', 'currentRatio', 'quickRatio','financialLeverage','capExToSales', 'priceToCashFlow',
        'epsGrowth', 'revenueGrowth', 'pegRatio', 'sharpeRatio', 'sortinoRatio', 'volatility', 'beta', 'marketCorr',
        'treynorRatio'])