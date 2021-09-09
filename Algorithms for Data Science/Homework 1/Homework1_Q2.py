import numpy as np
import pandas as pd


# Node class 
# The code for the HashNode class and the Hash table class were mostly from the site below but not all
# https://gist.github.com/Tetsuya3850/fe841bf1f1088fe1f804c189db4c9daf

class hashNode:  
    # Function to initialize the node object 
    def __init__(self, key, value): 
        self.next = None
        self.key = key
        self.value = value
   
class HashTable:
    def __init__(self):
        self.size = 150
        self.hashTable = [None] * self.size

    def hash(self, key):
        # Generate hash from key.
        # Time O(N), Space O(1), where N is the length of key.
        return key % self.size

    def add(self, key, value):
        # Add key, value.
        # Time O(1), Space O(1), where N is the num of elements in hashtable.
        key_hash = self.hash(key)
        if not self.hashTable[key_hash]:
            self.hashTable[key_hash] = hashNode(key, value)
        else:
            temp = self.hashTable[key_hash]
            while temp.next:
                temp = temp.next
            temp.next = hashNode(key, value)

    def printTable(self): 

        key = 0
        while key < self.size:
            print("")
            temp = self.hashTable[key]
            if temp != None:
                while temp:
                    print("Original Key & (Node key)%150: ", temp.key, temp.key%150)
                    print("Node Value: ", temp.value)
                    temp = temp.next
            key += 1


    def remove20Percent(self): 
        i = 0 # counter for number deleted
        n = int(.2*self.size) #size of array * 20% 
        print("n being removed", n)
        while i < n: #until n number deleted
            print("i: ", i)
            key = np.random.randint(150) #generate a random number from 0 to 150
            temp = self.hashTable[key] #find the hash value
            if temp != None: #if the position is not null 
                print("Removing full list at key: " ,key)
                self.hashTable[key] = None #set to null 
                i += 1 #increment deleted


def toHashTable(IrisMatrix):
    IrisHashTable = HashTable() # makes a Hashtable of size 150 see hashtable implementation

    index = 0 #set the row iteration
    i = 0 #set the column iteration
    while index < len(IrisMatrix): #rows
        while i<=4: #columns
            key = len(IrisMatrix)*i + index #unique key that would address to the same hash value
            print("i",i)
            print("key",i)
            IrisHashTable.add(key, IrisMatrix[index][i]) #add to hashTable
            print("IrisMatrix[index][i]", IrisMatrix[index][i])
            i += 1 #increase column counter
        index +=1 #increase row counter
        i = 0#reset column counter for next row
        print("reset i", i )
        print("increase index", index)
    return IrisHashTable         
    
if __name__ == '__main__':
    Iris = pd.read_csv("iris.csv",header=0)
    IrisMatrix = Iris.to_numpy() #loading csv into 

    ## Question 2 ##
    # My implemenations 
    IrisHT = toHashTable(IrisMatrix) 
    IrisHT.printTable()
    IrisHT.remove20Percent()
    print(IrisHT.hashTable)

