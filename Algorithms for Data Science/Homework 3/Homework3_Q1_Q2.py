from math import pi, sqrt
import math
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

def EMalgorithm(x, K, conv):
    # Function takes in the dataset, number of clusters and thee precision you want to converge with 10^-conv
    D = len(x[0]) # the Dimensions/# of features
    N = len(x)  # number of observations
    # Initializing Parameters
    x_col = np.matrix(np.mean(x, axis=0)).T # mean of column/feature
    sd_col = np.matrix(np.std(x, axis=0, ddof=1)).T # std dev of column/feature
    randn = np.random.randn(1,K) # 1xK random array
    # compute Initial Values
    mu_k = np.array(x_col*np.ones(K) + np.matmul(sd_col,randn))
    sd_k = np.mean(sd_col)*np.ones(K)
    p_k = [1/K]*K


    p_kn_1 = 0 #previous place holder for convergence 
    condition = True
    i = 0
    while condition:
        print()
        print("iteration: ", i+1)
        p_kn = E_Step(x,K, D, mu_k,sd_k, p_k) # run the E_step Function
        print("p_kn: ",p_kn)
        condition = np.any(np.abs((p_kn-p_kn_1))>(10**-conv)) #update the loop condition
        p_kn_1 = p_kn # update the prev matrix
        mu_k, sd_k, p_k = M_Step(x,p_kn, K,D,N) # run the M_step Function
        print("mu_k: ",mu_k)
        print("sd_k: ",sd_k)
        print("p_k: ", p_k)
        i+=1
    
    return 

def E_Step(x, K, D, mu_k, sd_k, p_k):
    # Function takes dataset, # of clusters, Dimensions, mean of cluster, std dev of cluster, p_k
    
    g = [None]*K 
    k = 0 # cluster counter
    while k < K:
        g[k] = (x-mu_k[:,k])
        g[k] = np.linalg.norm(g[k],axis=1)**2
        g[k] = -1/2* (g[k]/(sd_k[k]**2))
        g[k] = np.exp(g[k])
        g[k] = 1/(((np.sqrt(2*math.pi))*sd_k[k]))**D * g[k]
        k+=1
    g = np.asarray(g) #format to array
   
    k=0
    p_kn =[None]*K 
    while k<K:
        p_kn[k] = p_k[k]*g[k]
        k+=1
    p_kn = np.asarray(p_kn)
    
    p_kn = p_kn / np.sum(p_kn, axis=0)

    return p_kn

def M_Step(x,p_kn, K, D, N):
    # Function takes dataset, probability of observation being in a cluster matrix, # clusters, Dimensions, # of observations

    k = 0
    mu_i = [None]*K
    while k<K:
        mu_i[k] = np.matmul(p_kn[k],x)/np.sum(p_kn[k])
        k+=1
    mu_i= np.asarray(mu_i).T

    k = 0
    sd_i = [None]*K
    while k<K:
        norm = (x-mu_i[:,k])
        norm = np.linalg.norm(norm,axis=1)**2
        norm = np.sum(norm*p_kn[k])
        sd_i[k] = np.sqrt(1/D * norm/sum(p_kn[k]))
        k+=1

    p_i = 1/N * np.sum(p_kn, axis=1)
    
    return mu_i, sd_i, p_i


if __name__ == "__main__":
    Iris = pd.read_csv("iris.csv",header=0)
    IrisMatrix = Iris.to_numpy()
    IrisMatrix = IrisMatrix[:,0:4]
    IrisMatrix = IrisMatrix.astype(float)
    # print(IrisMatrix)

    x = np.array([[1,2],[4,2],[1,3],[4,3]])
    # run the Expectation maximization algorithm EMalgorithm(Data, # of clusters, 10^-precision value)
    # EMalgorithm(x,2,6)
    EMalgorithm(IrisMatrix,3,6)

    