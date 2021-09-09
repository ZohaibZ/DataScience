import numpy as np
import pandas as pd
from scipy import stats,linalg
import matplotlib.pyplot as plt

def plotAllbyFeatures(IrisMatrix, i, j):
    # quick plot function to display the 2 features together
    # Input is taking the IrisMatrix with the knowledge of each class being of size 50 and feature (i,j)
    index = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    plt.figure()
    plt.scatter(IrisMatrix[0:49,i], IrisMatrix[0:49,j], color='blue', label = 'Setosa') 
    plt.scatter(IrisMatrix[50:99,i], IrisMatrix[50:99,j], color='green',label = 'Versicolor')
    plt.scatter(IrisMatrix[100:149,i], IrisMatrix[100:149,j], color='orange',label = 'Virginica')
    plt.title('Displaying two Iris features.')
    plt.ylabel(index[j])
    plt.xlabel(index[i])
    plt.legend()
    plt.show()
    return 

def sortAllbyFeature(IrisMatrix, i ):
    # Input is taking the IrisMatrix along with feature to sort it by
    index = ['Sorted by sepal_length', 'Sorted by sepal_width', 'Sorted by petal_length', 'Sorted by petal_width']
    Sorted = IrisMatrix[IrisMatrix[:,i].argsort(kind ='mergsort')]
    print(index[i], Sorted)
    return Sorted

def outlierRemoval(IrisClass, t): 
    # function takes in one of the classes along with the Z score threshold to remove outliers observations
    threshold = t
    # calculate the absolute value of the Z score feature-column wise
    Z = np.abs(stats.zscore(IrisClass, axis=0, ddof=1))
    # find the indexes of where the Z values are greater than the threshhold
    remove = np.where(Z > threshold)
    # and delete the whole observation instead of having empty values in the observation
    # remove[0] stores a list of indices of the rows and remove[1] stores a list of indices of the column where
    # Z > threshhold
    IrisClass = np.delete(IrisClass, remove[0], axis=0)

    return IrisClass

def plotClassbyFeature(IrisClass, i, j):
    #quick plot function
    index = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    classIndex = ['Setosa', 'Versicolor', 'Virginica']
    plt.figure()
    plt.scatter(IrisClass[:,i], IrisClass[:,j], color='blue')
    plt.title('Iris Data')
    plt.ylabel(index[j])
    plt.xlabel(index[i])
    plt.show()
    return 

def plotAllClassbyFeature(IrisClass):
    #quick plot function
    index = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    fig = plt.figure()

    plt.subplot(3, 2, 1)
    plt.scatter(IrisClass[:,0], IrisClass[:,1], color='blue')
    plt.xlabel(index[0])
    plt.ylabel(index[1])
    
    plt.subplot(3, 2, 2)
    plt.scatter(IrisClass[:,0], IrisClass[:,2], color='green')
    plt.xlabel(index[0])
    plt.ylabel(index[2])

    plt.subplot(3, 2, 3)
    plt.scatter(IrisClass[:,0], IrisClass[:,3], color='orange')
    plt.xlabel(index[0])
    plt.ylabel(index[3])

    plt.subplot(3, 2, 4)
    plt.scatter(IrisClass[:,1], IrisClass[:,2], color='purple')
    plt.xlabel(index[1])
    plt.ylabel(index[2])

    plt.subplot(3, 2, 5)
    plt.scatter(IrisClass[:,1], IrisClass[:,3], color='red')
    plt.xlabel(index[1])
    plt.ylabel(index[3])
    
    plt.subplot(3, 2, 6)
    plt.scatter(IrisClass[:,2], IrisClass[:,3], color='brown')
    plt.xlabel(index[2])
    plt.ylabel(index[3])
    
    plt.show()

    return 

def fisherFeatureRanking(setosa, versicolor, virginica, i):
    #function takes in the 3 different classes, with the feature of interest
    fdr = [None]*3 #store the 3 different values in the array 
    # this is the numerator 
    corr_1= (np.mean(setosa[:,i]) - np.mean(versicolor[:,i]))**2
    # this is the denominator 
    s_1   = (np.std((setosa[:,i]), ddof=1)**2 + np.std((versicolor[:,i]), ddof=1))**2
    # then divide to get the value 
    fdr[0] = corr_1/s_1
    corr_2 = (np.mean(setosa[:,i]) - np.mean(virginica[:,i]))**2
    s_2   = (np.std((setosa[:,i]), ddof=1)**2 + np.std((virginica[:,i]), ddof=1))**2
    fdr[1] = corr_2/s_2
    corr_3 = (np.mean(versicolor[:,i]) - np.mean(virginica[:,i]))**2
    s_3   = (np.std((versicolor[:,i]), ddof=1)**2 + np.std((virginica[:,i]), ddof=1))**2
    fdr[2] = corr_3/s_3
    
    return fdr, sum(fdr)


if __name__ == '__main__':
    Iris = pd.read_csv("iris.csv",header=0)
    IrisMatrix = Iris.to_numpy()
    
    # IrisMatrix = np.array(IrisMatrix)

    setosa = np.array(IrisMatrix[0:50,0:4],dtype=np.float64)
    versicolor = np.array(IrisMatrix[50:100,0:4],dtype=np.float64)
    virginica = np.array(IrisMatrix[100:150,0:4],dtype=np.float64)

    sepal_l = IrisMatrix[:,0]
    sepel_w = IrisMatrix[:,1]
    petal_l = IrisMatrix[:,2]
    petal_w = IrisMatrix[:,3]
    # [0 = sepal length, 1 = sepal width, 2 = petal length, 3 = petal width]

    # plotAllbyFeatures(IrisMatrix, 0, 1)
    # plotAllbyFeatures(IrisMatrix, 0, 2)
    # plotAllbyFeatures(IrisMatrix, 0, 3)
    # plotAllbyFeatures(IrisMatrix, 1, 2)
    # plotAllbyFeatures(IrisMatrix, 1, 3)
    # plotAllbyFeatures(IrisMatrix, 2, 3)

    # sortedbysepal_l = sortAllbyFeature(IrisMatrix, 0)
    # sortedbysepal_w = sortAllbyFeature(IrisMatrix, 1)
    # sortedbypetal_l = sortAllbyFeature(IrisMatrix, 2)
    # sortedbypetal_w = sortAllbyFeature(IrisMatrix, 3)

    # 99 % coverage under the normal distribution
    # setosaOutlierRemoved = outlierRemoval(setosa,3)
    # print("setosa size after removal: ", len(setosaOutlierRemoved))
    # versicolorOutlierRemoved = outlierRemoval(versicolor,3)
    # print("versicolor size after removal: ", len(versicolorOutlierRemoved))
    # virginicaOutlierRemoved = outlierRemoval(virginica,3)
    # print("virginica size after removal: ", len(virginicaOutlierRemoved))

    # [0 = Setosa, 1 = Versicolor, 2 = Virginica] 
    # plotClassbyFeature(setosa,0,3)
    # plotClassbyFeature(setosaOutlierRemoved,0,3)
    # plotAllClassbyFeature(setosa)
    # plotAllClassbyFeature(setosaOutlierRemoved)

    # [0 = sepal length, 1 = sepal width, 2 = petal length, 3 = petal width]
    sepallength_fdr = fisherFeatureRanking(setosa, versicolor, virginica, 0)
    print()
    print("Sepal Length: ", sepallength_fdr)
    sepalwidth_fdr = fisherFeatureRanking(setosa, versicolor, virginica, 1)
    print()
    print("Sepal Width: ", sepalwidth_fdr)
    petallength_fdr = fisherFeatureRanking(setosa, versicolor, virginica, 2)
    print()
    print("Petal_Length: ", petallength_fdr)
    petalwidth_fdr = fisherFeatureRanking(setosa, versicolor, virginica, 3)
    print()
    print("Petal_Width: ", petalwidth_fdr)


    