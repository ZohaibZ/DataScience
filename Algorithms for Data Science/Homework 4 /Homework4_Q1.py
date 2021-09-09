import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from itertools import cycle

# Problem solved using the links below as refernce
# https://github.com/knathanieltucker/bit-of-data-science-and-scikit-learn/blob/master/notebooks/DensityEstimation.ipynb

def parzen_window(X,y,bw,i,j=None): 
    categories = y.unique()

    if j == None:
        X = (X.iloc[:,i]).values.reshape(-1,1)
        X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
        theta = np.linspace(0, max(X_train)+1, 1000).reshape(-1,1)
        estimators = []
        for c in categories:
            m = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_train[Y_train == c])
            estimators.append(m)

        calc_estimators(estimators, theta, X_test, Y_test, categories,j)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
        X1_train = (X_train.iloc[:,i]).values.reshape(-1,1)
        X2_train = (X_train.iloc[:,j]).values.reshape(-1,1)
        X1_test = (X_test.iloc[:,i]).values.reshape(-1,1)
        X2_test = (X_test.iloc[:,j]).values.reshape(-1,1)
        Ay, Ax = np.meshgrid(np.linspace(0,max(X1_train)+1,101), np.linspace(0,max(X2_train)+1,101))
        theta = np.vstack([Ay.ravel(), Ax.ravel()]).T
        print(theta.shape)
        xy_test = np.hstack([X1_test,X2_test])
        print(xy_test.shape)

        estimators = []
        for c in categories:
            xy_train = np.hstack([X1_train[Y_train == c],X2_train[Y_train == c]])
            m = KernelDensity(kernel='gaussian', bandwidth=bw).fit(xy_train)
            estimators.append(m)
        
        calc_estimators(estimators, theta, xy_test, Y_test, categories,j, Ax, Ay)

    return

def calc_estimators(estimators, theta, x_test, y_test,categories,j, Ax=None, Ay=None):
    
    if j == None:
        cycol = cycle('rgbcmykw')
        test_estimates = []
        for i in range(len(estimators)):
            log_dens = estimators[i].score_samples(theta)
            log_test = estimators[i].score_samples(x_test)
            test_estimates.append(np.exp(log_test))
            color=next(cycol)
            plt.plot(theta, np.exp(log_dens), c = color)
        

        classify = np.argmax((np.array(test_estimates).T),axis=1)
        category = []
        for c in classify:
            category.append(categories[c])
        
        print("classify: ", classify)
        print("y_test: ", y_test)
        print("category: ", category)
        print("category==y_test: ", category==y_test)
        plt.show()

        
    else:
        cycol = cycle(['Reds','Greens','Blues','Yellows','Browns','Oranges'])
        test_estimates = []
        for i in range(len(estimators)):
            log_dens = estimators[i].score_samples(theta)
            log_test = estimators[i].score_samples(x_test)
            f = np.reshape(np.exp(log_dens), Ax.shape)
            test_estimates.append(np.exp(log_test))
            color=next(cycol)
            plt.contour(Ax, Ay, f , cmap = color)


        classify = np.argmax((np.array(test_estimates).T),axis=1)
        category = []
        for c in classify:
            category.append(categories[c])
        
        print("classify: ", classify)
        print("y_test: ", y_test)
        print("category: ", category)
        print("category==y_test: ", category==y_test)
        plt.show()


    return test_estimates, classify


if __name__ == "__main__":
    Iris = pd.read_csv("iris.csv",header=0)
    X = Iris.iloc[:,0:4]
    y = Iris.iloc[:,4]
    
    # parzen_window(X,y,.5,2)
    parzen_window(X,y,.5,2,3)
    
    

    
    
    

    
