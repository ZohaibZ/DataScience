import numpy as np
import pandas as pd


class HashTable:  
    def __init__(self):
        self.size = 150
        self.hashTable = [[] for i in range(self.size)]
        
    def get_hash(self, key):
        return key % self.size
    
    def __getitem__(self, key):
        key_hash = self.get_hash(key)
        for kv in self.hashTable[key_hash]:
            if kv[0] == key:
                return kv[1]
            
    def __setitem__(self, key, val):
        key_hash = self.get_hash(key)
        found = False
        for idx, element in enumerate(self.hashTable[key_hash]):
            if len(element)==2 and element[0] == key:
                self.hashTable[key_hash][idx] = (key,val)
                found = True
        if not found:
            self.hashTable[key_hash].append((key,val))
        
    def __delitem__(self, key):
        key_hash = self.get_hash(key)
        for index, kv in enumerate(self.hashTable[key_hash]):
            if kv[0] == key:
                print("del",index)
                del self.hashTable[key_hash][index]
                
    
if __name__ == '__main__':
    Iris = pd.read_csv("iris.csv",header=0)
    IrisMatrix = Iris.to_numpy()