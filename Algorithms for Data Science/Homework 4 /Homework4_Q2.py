import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions

# Problem solved using the links below as refernce
# https://www.machinecurve.com/index.php/2020/05/03/creating-a-simple-binary-svm-classifier-with-python-and-scikit-learn/
# https://scikit-learn.org/stable/modules/svm.html

def gaussian_svm(X,y,i,j,C,gamma):
    X_train, X_test, y_train, y_test = train_test_split(X[:,[i,j]], y, test_size = 0.20, random_state = 0)

    # Fitting Kernel SVM to the Training set
    classifier = SVC(kernel = 'rbf', random_state = 0, C=C, gamma=gamma)
    classifier.fit(X_train, y_train)
    confidence = classifier.score(X_test, y_test)
    print(confidence)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(y_pred)
    # Get support vectors
    support_vectors = classifier.support_vectors_
    print(support_vectors)
    # Visualize support vectors
    plt.scatter(X_train[:,0], X_train[:,1])
    plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
    plt.title('Linearly separable data with support vectors')
    plt.xlabel('X1')
    plt.ylabel('X2')
    # Plot decision boundary
    plot_decision_regions(X_test, y_test, clf=classifier, legend=2, X_highlight=X_test)
    plt.show()

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred,average="weighted"))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred,average="weighted"))
    
    return classifier

if __name__ == "__main__":
    Iris = pd.read_csv("iris.csv",header=0)
    X = np.array(Iris.iloc[:,0:4])
    y = np.array(Iris['species'].astype('category').cat.codes)

    gaussian_svm(X,y,2,3,1,"auto")
