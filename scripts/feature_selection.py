import sys, datetime, pdb, time
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from algorithms.SBS import SBS
from utils.ml_utils import plot_decision_regions, standardize, IMG_PATH


# Sequential Backward Selection
def sbs_run(df, xcols, k_feats=5, est=KNeighborsClassifier(n_neighbors=3)):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=ts, random_state=0)
    
    # selecting features
    sbs = SBS(est, k_features=k_feats)
    sbs.fit(X_train, y_train)
    
    # plotting performance of feature subsets
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'sbs.png', dpi=300)
    
    pdb.set_trace()
    k5 = list(sbs.subsets_[10])
    print(df.columns[k5])
    
    est.fit(X_train, y_train)
    print('Training accuracy:', est.score(X_train, y_train))
    print('Test accuracy:', est.score(X_test, y_test))
    
    est.fit(X_train[:, k5], y_train)
    print('Training accuracy:', est.score(X_train[:, k5], y_train))
    print('Test accuracy:', est.score(X_test[:, k5], y_test))

def random_forest_feature_importance(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    feat_labels = df[list(xcols)].columns
    forest = RandomForestClassifier(n_estimators=10000,
                                  random_state=0,
                                  n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
            feat_labels[indices[f]], 
            importances[indices[f]]))
                    
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), 
                importances[indices],
                color='lightblue', 
                align='center')
  
    plt.xticks(range(X_train.shape[1]), 
             feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'random_forest_feat.png', dpi=300)
    
    X_selected = forest.transform(X_train, threshold=0.15)
    print(X_selected.shape)
    
def principal_component_analysis(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
          
    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('Eigenvalues \n%s' % eigen_vals)
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
         label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid',
          label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'pca1.png', dpi=300)
    plt.close()
    # plt.show()
    
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
    # print('Matrix W:\n', w)
    
    X_train_pca = X_train.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train.values==l, 0], 
                    X_train_pca[y_train.values==l, 1], 
                    c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'pca2.png', dpi=300)
    
def pca_scikit(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lr = lr.fit(X_train_pca, y_train)

    plot_decision_regions(X_train_pca, y_train.values, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'pca3.png', dpi=300)
    plt.close()
    
    plot_decision_regions(X_test_pca, y_test.values, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'pca4.png', dpi=300)
    # plt.show()
    
def linear_discriminant_analysis(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    np.set_printoptions(precision=4)
    mean_vecs = []
    y_set = list(y.unique())
    for label in y_set:
        mean_vecs.append(np.mean(X_train[y_train.values==label], axis=0))
        # print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
    d = len(xcols) # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(y_set, mean_vecs):
        class_scatter = np.zeros((d, d)) # scatter matrix for each class
        for row in X_train[y_train.values == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1) # make column vectors
            class_scatter += (row-mv).dot((row-mv).T)
        S_W += class_scatter                             # sum class scatter matrices
    print('Within-class scatter matrix: %s' % (S_W))
    print('Class label distribution: %s' % np.bincount(y_train))
    
    S_W = np.zeros((d, d))
    for label,mv in zip(y_set, mean_vecs):
        class_scatter = np.cov(X_train[y_train.values==label].T)
        S_W += class_scatter
    print('Scaled within-class scatter matrix: %s' % (S_W))
    
    mean_overall = np.mean(X_train, axis=0)
    d = len(xcols) # number of features
    S_B = np.zeros((d, d))
    for i,mean_vec in enumerate(mean_vecs):
        n = X_train[y_train==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1) # make column vector
        mean_overall = mean_overall.reshape(d, 1) # make column vector
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print('Between-class scatter matrix: %s' % (S_B))
    
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in decreasing order:\\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])
    
    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)
    
    plt.bar(range(0, d), discr, alpha=0.5, align='center',
            label='individual \"discriminability\"')
    plt.step(range(0, d), cum_discr, where='mid',
             label='cumulative \"discriminability\"')
    plt.ylabel('\"discriminability\" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'lda1.png', dpi=300)
    plt.close()
    
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                          eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\\n', w)
    
    X_train_lda = X_train.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train.values==l, 0] * (-1), 
                    X_train_lda[y_train.values==l, 1] * (-1), 
                    c=c, label=l, marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'lda2.png', dpi=300)

def lda_scikit(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    
    plot_decision_regions(X_train_lda, y_train.values, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'lda_scikit.png', dpi=300)
    plt.close()
    
    X_test_lda = lda.transform(X_test)
    
    plot_decision_regions(X_test_lda, y_test.values, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'lda_scikit_test.png', dpi=300)
    