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

from utils.ml_utils import plot_decision_regions, standardize, IMG_PATH

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