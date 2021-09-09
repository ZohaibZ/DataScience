import numpy as np
import pandas as pd
from scipy import stats,linalg
from scipy.fft import dct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# https://www.geeksforgeeks.org/python-replace-nan-values-with-average-of-columns/
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
# https://scikit-learn.org/stable/modules/naive_bayes.html

def dataCleanse(matrix):
    col_mean = np.nanmean(matrix, axis = 0) 
    # printing column mean 
    # find indices where nan value is present 
    inds = np.where(np.isnan(matrix)) 
    # replace inds with avg of column 
    matrix[inds] = np.take(col_mean, inds[1]) 
    # printing final array 

    return np.round(matrix,1)

def featureGenerator(matrix,n):
    features = np.zeros((n,2))

    i = 0
    while i < n:
        row = matrix[i] # extract each row
        rowDCT = dct(row, norm='ortho')
        features[i,:]= [np.mean(rowDCT), np.std(rowDCT,ddof=1)]
        i+=1
    return np.round(features,1)

def outlierRemoval(matrix, t): 
    # function takes in one of the classes along with the Z score threshold to remove outliers observations
    threshold = t
    # calculate the absolute value of the Z score feature-column wise
    Z = np.abs(stats.zscore(matrix, axis=0, ddof=1))
    # find the indexes of where the Z values are greater than the threshhold
    remove = np.where(Z > threshold)
    # and delete the whole observation instead of having empty values in the observation
    # remove[0] stores a list of indices of the rows and remove[1] stores a list of indices of the column where
    # Z > threshhold
    newMatrix = np.delete(matrix, remove[0], axis=0)

    return newMatrix

def plotClass(class1,class2,class3, i, j):
    #quick plot function
    index = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','f1','f2','f3','f4']
    plt.figure()
    plt.scatter(class1[:,i], class1[:,j], color='blue', label = 'setosa')
    plt.scatter(class2[:,i], class2[:,j], color='green',label = 'versicolor')
    plt.scatter(class3[:,i], class3[:,j], color='orange',label = 'virginica')
    plt.title('Iris Data')
    plt.ylabel(index[j])
    plt.xlabel(index[i])
    plt.legend()
    plt.show()
    return 

def topFeatures(X,Y):
    # feature extraction
    test = SelectKBest(score_func=f_classif, k=2)
    features = test.fit_transform(X, Y)
    # summarize selected features
    print("test.scores_: ", test.scores_)
    print("test.scores_.shape: ", test.scores_.shape)
    print("features.shape: ", features.shape)
    print("features: \n",features)

    return features

def classPCA(data):
    kpca = KernelPCA(n_components = 2, kernel = 'rbf')
    # scaler = StandardScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)
    print(data.shape)
    class_kpca = kpca.fit_transform(data)
    print("class_kpca.shape: ", class_kpca.shape)
    print("kpca.lambdas_: ", kpca.lambdas_)
    print("kpca.alphas_:", kpca.alphas_)

    plt.scatter(class_kpca[:, 0], class_kpca[:, 1], edgecolor='none', alpha=0.5)
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
    return class_kpca

def bayesClassifier(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    model = GaussianNB()
    model.fit(X_train, y_train)
    confidence = model.score(X_test, y_test)
    print("confidence: ", confidence)
    
    print("model params: \n", model.get_params())
    y_pred = model.predict(X_test)

    print("y_test: \n", y_test)
    print("y_pred: \n", y_pred)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred, average = None))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred, average = None))
    
    return model

def LDAClassifier(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    confidence = model.score(X_test, y_test)
    print("confidence: ", confidence)
    
    print("model params: \n", model.get_params())
    y_pred = model.predict(X_test)

    print("y_test: \n", y_test)
    print("y_pred: \n", y_pred)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred, average = None))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred, average = None))
    
    return model



def gaussian_svm(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

    # Fitting Kernel SVM to the Training set
    model = SVC(kernel = 'rbf')
    model.fit(X_train, y_train)
    confidence = model.score(X_test, y_test)
    print("confidence: ", confidence)

    print("model params: \n", model.get_params())
    y_pred = model.predict(X_test)

    print("y_test: \n", y_test)
    print("y_pred: \n", y_pred)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred, average = None))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred, average = None))
    
    return model 




if __name__ == '__main__':
    Iris = pd.read_csv("iris_6_features_for_cleansing.csv",header=0)

    # print(Iris)
    IrisMatrix = Iris.to_numpy()

    # segregate the data per class
    setosa = np.array(IrisMatrix[0:50,0:6],dtype=np.float64)
    versicolor = np.array(IrisMatrix[50:100,0:6],dtype=np.float64)
    virginica = np.array(IrisMatrix[100:150,0:6],dtype=np.float64)

    # Data Cleanse - Get everything in the same format and remove Null observations
    setosa = dataCleanse(setosa)
    # print(setosa)
    versicolor = dataCleanse(versicolor)
    # print(versicolor)
    virginica = dataCleanse(virginica)
    # print(virginica)

    Iris = np.vstack((setosa,versicolor,virginica))
    X = Iris
    y = IrisMatrix[:,6]

    # # segregate the data per feature
    # sepal_l = IrisMatrix[:,0]
    # sepel_w = IrisMatrix[:,1]
    # petal_l = IrisMatrix[:,2]
    # petal_w = IrisMatrix[:,3]
    # feature1 = IrisMatrix[:,4]
    # feature2 = IrisMatrix[:,5]

    # # feature generation
    # setosaFeatures = featureGenerator(setosa,50)
    # # print("setosa features: ", setosaFeatures)
    # versicolorFeatures = featureGenerator(versicolor,50)
    # # print("versicolor features: ", versicolorFeatures)
    # virginicaFeatures = featureGenerator(virginica,50)
    # # print("virginica features: ", virginicaFeatures)

    # # join the class with new features
    # newSetosa = np.hstack((setosa,setosaFeatures))
    # print("New setosa class: \n", newSetosa)
    # newVersicolor = np.hstack((versicolor,versicolorFeatures))
    # print("versicolor features: \n", newVersicolor)
    # newVirginica = np.hstack((virginica,virginicaFeatures))
    # print("virginica features: \n", newVirginica)

    # # 99% coverage under the normal distribution
    # setosaOutliersRemoved = outlierRemoval(setosa, 2.576)
    # print("setosa size after removal: ", len(setosaOutliersRemoved))
    # versicolorOutliersRemoved = outlierRemoval(versicolor, 2.576)
    # print("versicolor size after removal: ", len(versicolorOutliersRemoved))
    # virginicaOutliersRemoved = outlierRemoval(virginica, 2.576)
    # print("virginica size after removal: ", len(virginicaOutliersRemoved))

    # # plot features over all classes
    # plotClass(newSetosa,newVersicolor,newVirginica, 3,6)

    # # Feature Rank
    # topFeature = topFeatures(X,y)

    # compute PCA
    # setosakPCA = classPCA(setosa)
    # versicolorkPCA = classPCA(versicolor)
    # verginicakPCA = classPCA(virginica)

    # # Bayes Classifier
    # bayesModel = bayesClassifier(X,y)

    # # Linear Discriminant Analysis
    # ldaModel = LDAClassifier(X,y)

    # Gaussian SVM 
    svmModel = gaussian_svm(X,y)