import sys, datetime, pdb, time
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RANSACRegressor

from utils.ml_utils import plot_decision_regions, standardize, IMG_PATH, lin_regplot
from algorithms.linear_regression_gd import LinearRegressionGD


def heat_map(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    cols = ['target_proxy'] + list(xcols)
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    sns.set(style='whitegrid', context='notebook')    
    sns.pairplot(df[cols], size=2.5)    
    plt.tight_layout()    
    plt.savefig(IMG_PATH + 'corr_mat.png', dpi=300)
    plt.close()
    
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, 
                cbar=True,
                annot=True, 
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=cols,
                xticklabels=cols)
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'heat_map.png', dpi=300)
    plt.close()
    
def linear_regressor(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    lr = LinearRegressionGD()
    lr.fit(np.transpose(np.array([X_train])), y_train)
    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'cost.png', dpi=300)
    plt.close()
    
    lin_regplot(np.transpose(np.array([X_train])), y_train, lr)
    
    # Find the average return of a stock with PE = 20
    # Note: will give odd results if x values are standardized and input is not
    y_val_std = lr.predict([20.0])
    print("Estimated Return: %.3f" % y_val_std)
    print('Slope: %.3f' % lr.w_[1])
    print('Intercept: %.3f' % lr.w_[0])


def linear_regression_sklearn(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    X = np.transpose(np.array([X]))      
    slr = LinearRegression()
    slr.fit(X, y.values)
    y_pred = slr.predict(X)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)
    
    lin_regplot(X, y.values, slr)
    plt.xlabel('x val')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'scikit_lr_fit.png', dpi=300)
    plt.close()

    # Closed-form solution
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    print('Slope: %.3f' % w[1])
    print('Intercept: %.3f' % w[0])