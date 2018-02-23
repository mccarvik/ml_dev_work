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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from algorithms.perceptron import Perceptron
from algorithms.adalinegd import AdalineGD
from algorithms.adalinesgd import AdalineSGD
from utils.ml_utils import plot_decision_regions, standardize, IMG_PATH, IMG_ROOT


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
    
def logisticRegression(df, xcols, C=100, penalty='l2'):
    # Need xcols to be a tuple for the timeme method to work VERY HACKY
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=ts, random_state=0)
    
    # Normalization of the data --> max = 1, min=0, etc
    # mms = MinMaxScaler()
    # X_train_norm = mms.fit_transform(X_train)
    # X_test_norm = mms.transform(X_test)
    
    # C: regularization parameter, (C = 1/lambda)
    # smaller C = more regulatiazion, smaller wieghts,  higher C = less regularization, lareger weights   
    # penalty: type of regulatizaion function used for weight shrinkage / decay to prevent overfitting
    lr = LogisticRegression(C=C, random_state=0, penalty=penalty)
    lr.fit(X_train, y_train)
    
    # Shows the percentage of falling into each class
    print("Class breakdowns: " + str(lr.predict_proba(X_test[0:1])))
    print('Training accuracy:', lr.score(X_train, y_train))
    print('Test accuracy:', lr.score(X_test, y_test))
    print("y-intercept:" + str(lr.intercept_))
    print("coeffs:" + str(lr.coef_))
    
    try:
        plot_decision_regions(X.values, y_train.values, classifier=lr)
        plt.title('Logistic Regression')
        plt.xlabel(list(X.columns)[0])
        plt.ylabel(list(X.columns)[1])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(IMG_ROOT + 'dow/log_reg_1.png', dpi=300)
        plt.close()
    except Exception as e:
        print("May have more than 2 variables")

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
    pdb.set_trace()
    y = df['target']
    X = df[list(xcols)]
    
    # Standardize and split the training and test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=ts, random_state=0)
    
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
    plt.savefig(IMG_ROOT + 'snp/kmeans/kkn.png', dpi=300)
    plt.close()