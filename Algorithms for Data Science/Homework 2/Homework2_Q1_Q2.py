import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 

# Help taken from these websites below on how to apply the Skewness and kurtosis
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
# including some parts of the algorithm from the matlab 
# syntheticDataExample.m file provided in the course material

def statistics(IrisMatrix): 
    # input of Numpy Matrix of a specific class
    # initializing my variables 
    # all as 1x4 each position representing the features as follow 
    # [sepal length, sepal width, petal length, petal width]
    minimum = [None]*4 
    maximum = [None]*4
    avg = [None]*4
    stddev = [None]*4
    skewness = [None]*4
    kurtosis = [None]*4
    trimmedmean1 = [None]*4
    trimmedmean2 = [None]*4
    trimmedmean3 = [None]*4
    # computed the covariance of the class [50x4] before sorting each column of the classes for the trimmed mean 
    # sorting the data will cause the covariance to be different as it will produce different pairs of data that were not originally there 
    covar = np.cov(IrisMatrix.astype(float),rowvar=False)
    # Now sort the data by column to compute the trimmed mean 
    IrisMatrix = np.sort(IrisMatrix, axis=0)

    i=0 # Initialize column counter
    while i < 4:
        minimum[i] = min(IrisMatrix[:,i]) # minimum of column
        maximum[i] = max(IrisMatrix[:,i]) # maximum of column
        avg[i] = np.mean((IrisMatrix[:,i])) # mean of column
        avg[i] = np.round(avg[i], 4) # round the mean to 4 decimals
        #compute the trimmed mean of p=1
        trimmedmean1[i] = np.mean((IrisMatrix[1:len(IrisMatrix)-1,i])) 
        trimmedmean1[i] = np.round(trimmedmean1[i], 7) # 7 decimals
        trimmedmean2[i] = np.mean((IrisMatrix[2:len(IrisMatrix)-2,i])) # p=2
        trimmedmean2[i] = np.round(trimmedmean1[i], 7) 
        trimmedmean3[i] = np.mean((IrisMatrix[3:len(IrisMatrix)-3,i])) # p=3
        trimmedmean3[i] = np.round(trimmedmean1[i], 7) 
        # compute standard deviation over n-1 factor
        stddev[i] = np.std((IrisMatrix[:,i]), ddof=1) 
        stddev[i] = np.round(stddev[i], 4)
        # compute skewness 
        skewness[i] = stats.skew(IrisMatrix[:,i])
        skewness[i] = np.round(skewness[i], 4)
        # compute kurtosis using the pearson definition
        kurtosis[i] = stats.kurtosis(IrisMatrix[:,i], fisher=False)
        kurtosis[i] = np.round(kurtosis[i], 4)
        i += 1 # Increment column counter

    return ("Minimum: ",minimum, "Maximum: ",maximum, "Mean: ",avg, "Trimmed Mean(p=1): ",trimmedmean1, "Trimmed Mean(p=2): ",trimmedmean2,"Trimmed Mean(p=3): ",trimmedmean3,"StdDev: ",stddev,"Covariance: ",covar, "Skewness: ", skewness, "Kurtosis: ", kurtosis)
    
def syntheticData(n,minimun, maximum, avg, covar): 
    # takes n = number of synthetic data points wanted, minimum, maximum, mean, and a numpy covariance matrix as input
    # make a nx4 uniform distributed random data matrix
    randData = np.random.rand(n,4) 
    # matrix multiplication of the random data with covariance [4x4] matrix 
    randData = np.matmul(randData,covar)
    # store the sythenthetic data in an nx4 matrix
    syntheticData = np.zeros((n,4))
   #store into synthetic data matrix
    mu = [None]*4

    i=0 # column counter
    while i<4:
        # Find min max of rand data column
        randMin = min(randData[:,i])
        randMax = max(randData[:,i])
        # Find min max of Iris data column
        a = minimun[i]
        b = maximum[i]
        # Compute min-max normalization 
        X = (((randData[:,i] - randMin)/(randMax - randMin)) * (b-a)) + a
        #compute the difference between synthetic and original data mean
        mu[i] = np.mean(X) - avg[i]
        #store into synthetic data matrix after adjusting mean by the difference
        syntheticData[:,i] = X - mu[i]
        i+=1 #increment column counter

    return syntheticData

def plot(original, synthetic, i, j):
    #quick plot function
    index = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    plt.figure()
    plt.scatter(original[:,i], original[:,j], color='blue', label = 'Original')
    plt.scatter(synthetic[:,i], synthetic[:,j], color='red', label = 'Synthetic')
    plt.title('Synthetic vs. Iris Data')
    plt.ylabel(index[j])
    plt.xlabel(index[i])
    plt.legend()
    plt.show()
    return 

def plotAllClasses(o1,s1,o2,s2,o3,s3, i, j):
    #quick plot function
    index = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    plt.figure()
    plt.scatter(o1[:,i], o1[:,j], color='blue', label = 'setosa_org')
    plt.scatter(s1[:,i], s1[:,j], color='red', label = 'setosa_synth')
    plt.scatter(o2[:,i], o2[:,j], color='green',label = 'versicolor_org')
    plt.scatter(s2[:,i], s2[:,j], color='purple',label = 'versicolor_synth')
    plt.scatter(o3[:,i], o3[:,j], color='orange',label = 'virginica_org')
    plt.scatter(s3[:,i], s3[:,j], color='brown',label = 'virginica_synth')
    plt.title('Synthetic vs. Iris Data')
    plt.ylabel(index[j])
    plt.xlabel(index[i])
    plt.legend()
    plt.show()
    return 

if __name__ == '__main__':
    Iris = pd.read_csv("iris.csv",header=0)
    IrisMatrix = Iris.to_numpy()
    IrisMatrixCov = np.array(IrisMatrix[:,0:4])

    setosa = IrisMatrix[0:50,0:4]
    versicolor = IrisMatrix[50:100,0:4]
    virginica = IrisMatrix[100:150,0:4]
    
    sepal_l = IrisMatrix[:,0]
    sepel_w = IrisMatrix[:,1]
    petal_l = IrisMatrix[:,2]
    petal_w = IrisMatrix[:,3]


    # My Implentations

    setosa_stats = statistics(setosa)
    versicolor_stats = statistics(versicolor)
    virginica_stats = statistics(virginica)
    covar = np.cov(IrisMatrixCov.astype(float),rowvar=False)

    print()
    print(setosa_stats) 
    print() 
    print(versicolor_stats)
    print()
    print(virginica_stats)
    print()
    print(covar)
    print()

    # 0 = sepal length, 1 = sepal width, 2 = petal length, 3 = petal width
    setosa_synthetic= syntheticData(100,setosa_stats[1],setosa_stats[3],setosa_stats[5], setosa_stats[15])
    np.savetxt("setosa_synthetic.csv", setosa_synthetic, delimiter=",",fmt='%5.4f')
    plot(setosa, setosa_synthetic, 0, 3)

    versicolor_synthetic = syntheticData(100,versicolor_stats[1],versicolor_stats[3],versicolor_stats[5], versicolor_stats[15])
    np.savetxt("versicolor_synthetic.csv", versicolor_synthetic, delimiter=",",fmt='%5.4f')
    plot(versicolor, versicolor_synthetic, 0, 3)
    
    virginica_synthetic= syntheticData(100,virginica_stats[1],virginica_stats[3],virginica_stats[5], virginica_stats[15])
    np.savetxt("virginica_synthetic.csv", virginica_synthetic, delimiter=",",fmt='%5.4f')
    plot(virginica, virginica_synthetic, 0, 3)

    plotAllClasses(setosa, setosa_synthetic,versicolor, versicolor_synthetic,virginica, virginica_synthetic,0,3)

