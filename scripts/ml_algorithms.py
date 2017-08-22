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
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import Perceptron as perceptron_skl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.grid_search import GridSearchCV

from algorithms.perceptron import Perceptron
from algorithms.adalinegd import AdalineGD
from algorithms.adalinesgd import AdalineSGD
from algorithms.SBS import SBS
from utils.ml_utils import plot_decision_regions, standardize


IMG_PATH = '/home/ubuntu/workspace/ml_dev_work/static/img/'


def run_perceptron(df, xcols, eta=0.1, n_iter=10):
    ''' Takes the pruned dataframe and runs it through the perceptron class
    
        Parameters
        ==========
        df : dataframe
            dataframe with the inputs and target
        eta : float
            learning rate between 0 and 1
        n_iter : int
            passes over the training dataset
        
        Return
        ======
        NONE
    '''
    t0 = time.time()
    y = df['target']
    X = df[list(xcols)]
    
    buy = df[df['target'] > 0][list(X.columns)].values
    sell = df[df['target'] < 0][list(X.columns)].values
    plt.figure(figsize=(7,4))
    plt.scatter(buy[:, 0], buy[:, 1], color='blue', marker='x', label='Buy')
    plt.scatter(sell[:, 0], sell[:, 1], color='red', marker='s', label='Sell')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend()
    ppn = Perceptron(eta, n_iter)
    ppn.fit(X.values, y.values)
    # pdb.set_trace()
    plot_decision_regions(X.values, y.values, classifier=ppn)
    plt.savefig(IMG_PATH + "scatter.png")
    plt.close()
    
    # print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
    plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.savefig(IMG_PATH + "misclassifications.png")
    plt.close()
    
    t1 = time.time()
    app.logger.info("Done training data and creating charts, took {0} seconds".format(t1-t0))
    print("Done training data and creating charts, took {0} seconds".format(t1-t0))
    
def run_perceptron_multi(df, xcols, eta=0.1, n_iter=15):
    t0 = time.time()
    y = df['target']
    X = df[list(xcols)]
    
    # Split up the training and test data and standardize inputs
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
    X_train_std, X_test_std = standardize(X_train, X_test)

    # pdb.set_trace()
    strong_buy = df[df['target'] == 3][list(X.columns)].values
    buy = df[df['target'] == 2][list(X.columns)].values
    sell = df[df['target'] == 1][list(X.columns)].values
    strong_sell = df[df['target'] == 0][list(X.columns)].values
    
    plt.figure(figsize=(7,4))
    plt.scatter(buy[:, 0], buy[:, 1], color='blue', marker='x', label='Buy')
    plt.scatter(sell[:, 0], sell[:, 1], color='red', marker='s', label='Sell')
    plt.scatter(strong_buy[:, 0], strong_buy[:, 1], color='blue', marker='*', label='Strong Buy')
    plt.scatter(strong_sell[:, 0], strong_sell[:, 1], color='red', marker='^', label='Strong Sell')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend()
    
    ppn = perceptron_skl(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)

    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    plot_decision_regions(X.values, y.values, classifier=ppn)
    plt.savefig(IMG_PATH + "scatter.png")
    plt.close()
    
    t1 = time.time()
    # app.logger.info("Done training data and creating charts, took {0} seconds".format(t1-t0))
    print("Done training data and creating charts, took {0} seconds".format(t1-t0))

def adalinegdLearningExample(df, xcols, eta=0.1, n_iter=10):
    # Learning rate too high - overshoot global min
    y = df['target']
    X = df[list(xcols)]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdalineGD(n_iter=20, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    
    # Learning rate too low - takes forever
    ada2 = AdalineGD(n_iter=20, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    
    plt.tight_layout()
    plt.savefig(IMG_PATH + "adaline_1.png", dpi=300)
    plt.close()
    # plt.show()
    
def adalineSGD(df, xcols, eta=0.1, n_iter=10):
    y = df['target']
    X = df[list(xcols)]
    
    # standardize features
    X_std = np.copy(X.values)
    # X_std[:,0] = (X.values[:,0] - X.values[:,0].mean()) / X.values[:,0].std()
    # X_std[:,1] = (X.values[:,1] - X.values[:,1].mean()) / X.values[:,1].std()
    
    ada = AdalineSGD(n_iter=15, eta=0.001, random_state=1)
    # pdb.set_trace()
    ada.fit(X_std, y.values)
    # pdb.set_trace()
    # ada.partial_fit(X_std, y.values)
    
    plot_decision_regions(X_std, y.values, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'adalinesgd.png', dpi=300)
    plt.close()
    
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'adalinesgd_gd.png', dpi=300)
    plt.close()

def adalineGD(df, xcols, eta=0.1, n_iter=10):
    y = df['target']
    X = df[list(xcols)]
    
    # standardize features
    X_std = np.copy(X.values)
    # X_std[:,0] = (X.values[:,0] - X.values[:,0].mean()) / X.values[:,0].std()
    # X_std[:,1] = (X.values[:,1] - X.values[:,1].mean()) / X.values[:,1].std()
    
    ada = AdalineGD(n_iter=15, eta=0.00001)
    ada.fit(X_std, y)
    
    plot_decision_regions(X_std, y.values, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'adaline_2.png', dpi=300)
    plt.close()
    
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'adaline_3.png', dpi=300)
    plt.close()
    
def logisticRegression(df, xcols, C=100, penalty='l1'):
    # Need xcols to be a tuple for the timeme method to work VERY HACKY
    y = df['target']
    X = df[list(xcols)]
    X_std = standardize(X)
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=0.3, random_state=0)
    
    # Normalization of the data --> max = 1, min=0, etc
    # mms = MinMaxScaler()
    # X_train_norm = mms.fit_transform(X_train)
    # X_test_norm = mms.transform(X_test)
    
    # C is the regularization parameter, (C = 1/lambda) --> The larger lambda is, 
    # the more regularized the weights are, the less susceptible the regression is 
    # to overfitting aka the smaller C is, the more regularized
    lr = LogisticRegression(C=C, random_state=0, penalty=penalty)
    lr.fit(X_train, y_train)
    
    # Shows the percentage of falling into each class
    # Will need this later when we use on current data
    print(lr.predict_proba(X_test[0:1]))
    
    print('Training accuracy:', lr.score(X_train, y_train))
    print('Test accuracy:', lr.score(X_test, y_test))
    print(lr.intercept_)
    print(lr.coef_)
    
    # plot_decision_regions(X_std, y_train.values, classifier=lr)
    plt.title('Logistic Regression')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'log_reg_1.png', dpi=300)
    plt.close()

def support_vector_machines(df, xcols, C=100):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    svm = SVC(kernel='linear', C=C, random_state=0)
    svm.fit(X_train, y_train)
    
    print('Training accuracy:', svm.score(X_train, y_train))
    print('Test accuracy:', svm.score(X_test, y_test))
    
    # plot_decision_regions(X.values, y.values, classifier=svm, test_break_idx=int(len(y)*(1-ts)))
    plot_decision_regions(X_std, y.values, classifier=svm)
    plt.title('Support Vector Machines')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'svm_C' + str(C) + '.png', dpi=300)
    plt.close()
    
def nonlinear_svm(df, xcols, C=100, gamma=0.10):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=C)
    svm.fit(X_train, y_train)
    
    print('Training accuracy:', svm.score(X_train, y_train))
    print('Test accuracy:', svm.score(X_test, y_test))
    
    plot_decision_regions(X_std, y.values, classifier=svm)
    plt.title('Support Vector Machines - Non Linear')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'svm_nonlinear_C' + str(C) + '.png', dpi=300)
    plt.close()
    
def decision_tree(df, xcols, md=3):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=md, random_state=0)
    tree.fit(X_train, y_train)

    print('Training accuracy:', tree.score(X_train, y_train))
    print('Test accuracy:', tree.score(X_test, y_test))
    
    plot_decision_regions(X_std, y.values, classifier=tree)
    plt.title('Decision Tree')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'dec_tree' + '.png', dpi=300)
    plt.close()
    
    export_graphviz(tree, 
                  out_file='tree.dot', 
                  feature_names=list(xcols))
    
    # execute "dot -Tpng tree.dot -o tree.png" to turn file into png file

def random_forest(df, xcols, estimators=5):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    forest = RandomForestClassifier(criterion='entropy',
                                  n_estimators=estimators, 
                                  random_state=1,
                                  n_jobs=3)
    forest.fit(X_train, y_train)

    print('Training accuracy:', forest.score(X_train, y_train))
    print('Test accuracy:', forest.score(X_test, y_test))
    
    plot_decision_regions(X_std, y.values, classifier=forest)
    plt.title('Randaom Forest (Decision Tree Ensemble)')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'random_forest' + '.png', dpi=300)
    plt.close()
    
def k_nearest_neighbors(df, xcols, k=5):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train, y_train)

    print('Training accuracy:', knn.score(X_train, y_train))
    print('Test accuracy:', knn.score(X_test, y_test))
    
    plot_decision_regions(X_std, y.values, classifier=knn)
    plt.title('Randaom Forest (Decision Tree Ensemble)')
    plt.xlabel(list(X.columns)[0])
    plt.ylabel(list(X.columns)[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'kkn' + '.png', dpi=300)
    plt.close()
    
###################################
# Feature selection / Extraction
###################################

# Sequential Backward Selection
def sbs_run(df, xcols, k_feats=2, est=KNeighborsClassifier(n_neighbors=3)):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
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
    
    k5 = list(sbs.subsets_[10])
    print(df.columns[1:][k5])
    
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
    
###################################
# Model Evaluation
###################################

def kfold_cross_validation(df, xcols, folds=10):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('clf', LogisticRegression(random_state=1))])

    kfold = StratifiedKFold(y=y_train, 
                            n_folds=folds,
                            random_state=1)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train.values[train])
        score = pipe_lr.score(X_train[test], y_train.values[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train.values[train]), score))
    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    scores = cross_val_score(estimator=pipe_lr, 
                             X=X_train, 
                             y=y_train.values, 
                             cv=10,
                             n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
def learning_curves(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
        
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', random_state=0))])

    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, 
                                            X=X_train, y=y_train, 
                                            train_sizes=np.linspace(0.1, 1.0, 10), 
                                            cv=10, n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, 
                     train_mean + train_std,
                     train_mean - train_std, 
                     alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')
    plt.fill_between(train_sizes, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'learning_curve.png', dpi=300)
    plt.close()

def validation_curves(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
          
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', random_state=0))])
    
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
                    estimator=pipe_lr, 
                    X=X_train, 
                    y=y_train, 
                    param_name='clf__C', 
                    param_range=param_range,
                    cv=10)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')
    plt.plot(param_range, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')
    plt.fill_between(param_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'val_curve.png', dpi=300)
    plt.close()

def grid_search_analysis(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    pipe_svc = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=1))])
    
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    param_grid = [{'clf__C': param_range, 
                   'clf__kernel': ['linear']},
                     {'clf__C': param_range, 
                      'clf__gamma': param_range, 
                      'clf__kernel': ['rbf']}]
    
    gs = GridSearchCV(estimator=pipe_svc, 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Test accuracy: %.3f' % clf.score(X_test, y_test))
    
    gs = GridSearchCV(estimator=pipe_svc, 
                                param_grid=param_grid, 
                                scoring='accuracy', 
                                cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), 
                                param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], 
                                scoring='accuracy', 
                                cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    
    
    
    
    
    