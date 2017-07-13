#Import Library
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

# https://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/


def get_data():
    df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    return (y,X)

def support_vector_machines():
    # Create SVM classification object 
    y, X = get_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = SVC(kernel='linear', C=1, gamma=1) 
    model.fit(x_train, y_train)
    print(model.score(x_train, y_train))

    #Predict Output
    predicted= model.predict(x_test)
    print(predicted)
    
    # create a mesh to plot in
    C = 1.0 # SVM regularization parameter
    svc = SVC(kernel='linear', C=1,gamma=0).fit(x_train, y_train)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
     np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()
    

if __name__ == '__main__':
    support_vector_machines()