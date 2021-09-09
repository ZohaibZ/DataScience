import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def gaussianKernel(X1, X2, spread):
    print("X1:", X1)
    print()
    print("X2:",X2)
    row1,col1 = X1.shape
    row2,col2 = X2.shape
    N = col2
    D = row2
    K = np.zeros(col1,col2)
    for i in col1:
        for j in col2:
            K[i,j] = (1/N)*(1/(((2*math.pi)*spread))**(D/2))*math.exp(-0.5*(((X1[:,i]-X2[:,j]).T*(X1[:,i]-X2[:,j]))/(spread^2)))
    return K


def parzen_estimation(iris_data,j):

    ax = np.linspace(0,8,200)

    spread = 0.5
    p1=np.zeros(np.size(ax));  
    for i in range(1,50):
        p1 = p1 + gaussianKernel(iris_data[i,j], ax, spread)
    plt.plot(ax,p1,color="red")

    p2=np.zeros(np.size(ax));  
    for i in range(51,100):
        p2 = p2 + gaussianKernel(iris_data[i,j], ax, spread)
    plt.plot(ax,p1,color="green")

    p3=np.zeros(np.size(ax));  
    for i in range(101,150):
        p3 = p3 + gaussianKernel(iris_data[i,j], ax, spread)
    plt.plot(ax,p1,color="green")

    return p1,p2,p3

if __name__ == "__main__":
    Iris = pd.read_csv("iris.csv",header=0)
    print(Iris.head)
    IrisMatrix = Iris.to_numpy()
    IrisMatrix = IrisMatrix[:,0:4]

    X = parzen_estimation(IrisMatrix,3)
    print(X)