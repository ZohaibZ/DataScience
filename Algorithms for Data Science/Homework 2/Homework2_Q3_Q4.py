import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import dct
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 

# Help taken from these websites below on how to apply the DCT
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
# https://stackoverflow.com/questions/40104377/issiue-with-implementation-of-2d-discrete-cosine-transform-in-python
# including some parts of the algorithm from the matlab 
# numericalFeatureGeneratorExample.m file provided in the course material

def featureGenerator(trainMatrix, n):
    features = np.zeros((n,13))

    i = 0
    while i < n:
        img = np.reshape(trainMatrix[i,1:len(trainMatrix[0])], (28,28))
        imgDCT = dct((dct(img, norm='ortho')).T, norm='ortho')
        diagDCT = np.diag(imgDCT) #extract diagonal
        rowDCT = imgDCT[0,:] # extract first row
        colDCT = imgDCT[:,0] #extract first column
        features[i,:]= [np.mean(diagDCT), np.std(diagDCT,ddof=1), stats.skew(diagDCT), stats.kurtosis(diagDCT,fisher=False),
            np.mean(rowDCT), np.std(rowDCT,ddof=1), stats.skew(rowDCT), stats.kurtosis(rowDCT,fisher=False),
            np.mean(colDCT), np.std(colDCT,ddof=1), stats.skew(colDCT), stats.kurtosis(colDCT,fisher=False),
            int(trainMatrix[i][0])]
        i+=1
    return features

def plot(trainMatrix, index):
    sub = 1
    i=0
    plt.figure()
    while sub <= len(index):
        plt.subplot(3,4,sub)
        img = np.reshape(trainMatrix[index[i],1:len(trainMatrix[0])], (28,28))
        plt.imshow(img, cmap='gray')
        sub+=1
        i+=1

    plt.show()

    return 

if __name__ == '__main__':
    train = pd.read_csv("train.csv",header=0)
    trainMatrix = train.to_numpy()

    test = featureGenerator(trainMatrix,100)
    np.savetxt("test.csv", test, delimiter=",", fmt='%5.4f')
    
    x = [1, 2, 4, 7, 8, 9, 11, 12, 17, 22]
    plot(trainMatrix, x)


 
