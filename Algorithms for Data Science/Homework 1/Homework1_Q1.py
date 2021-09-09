import numpy as np
import pandas as pd

# Node class 
# The code for the Node Class and Linked List class are from these 2 websites below
# https://www.geeksforgeeks.org/linked-list-set-1-introduction/
# https://www.geeksforgeeks.org/linked-list-set-2-inserting-a-node/?ref=lbp

class Node:  
    # Function to initialize the node object 
    def __init__(self, data): 
        self.data = data  # Assign data 
        self.next = None  # Initialize  
                          # next as null 

# Linked List class 
class LinkedList: 
    # Function to initialize the Linked  
    # List object 
    def __init__(self):  
        self.head = None

    # This function is defined in Linked List class 
    # Appends a new node at the end.  This method is 
    # defined inside LinkedList class shown above */ 
    def append(self, new_data):
        # 1. Create a new node 
        # 2. Put in the data 
        # 3. Set next as None
        new_node = Node(new_data)

        # 4. If the Linked List is empty, then make the 
        #    new node as head 
        if self.head is None: 
            self.head = new_node
            return 

        # 5. Else traverse till the last node 
        last = self.head 
        while (last.next): 
            last = last.next

        # 6. Change the next of last node 
        last.next =  new_node 

    def printList(self): 
        temp = self.head 
        while (temp): 
            print(temp.data) 
            temp = temp.next

def toArrayofLL(IrisMatrix): #takes in a numpy matrix
    IrisLL = [0]*int(len(IrisMatrix)) # makes an array of zeroes for the size of the array
    index = 0 #set the row iteration
    i = 0  #set the column iteration
    while index < len(IrisMatrix): #iterating through each row
        IrisLL[index] = LinkedList() #making a new linked list
        print("IrisLL index",IrisLL[index]) 
        print("index", index)
        while i<=4: #iterating through each column position of the n - 1 row
            print("i",i)
            IrisLL[index].append(IrisMatrix[index][i]) #append to the end of the linked list
            print("IrisMatrix[index][i]", IrisMatrix[index][i])
            i += 1 #increase column counter
        index +=1 #increase row counter
        i = 0 #reset column counter 
        print("reset i", i )
        print("increase index", index)
    return IrisLL

def printAllLLinArray(IrisLL):
    pos = 0
    while pos < len(IrisLL):
        if IrisLL[pos] != 0:
            print("")
            IrisLL[pos].printList()
            print("")
        pos += 1

def remove20Percent(IrisLL): #takes in the array of linked lists 
    i = 0 #start count of deleted
    n = int(.2*len(IrisLL)) #calculate 20% of the size of the array
    print("n being removed", n)
    while i < n: # while not 20% 
        x = np.random.randint(len(IrisLL)) #random number from 0 to len(IrisLL array)
        if IrisLL[x] != 0: #if not already set to 0
            print("Removing" ,IrisLL[x].head.data)
            print("at x" ,x)
            IrisLL[x] = 0 #set the index to 0
            i += 1 #increase the i count

    
if __name__ == '__main__':
    Iris = pd.read_csv("iris.csv",header=0)
    IrisMatrix = Iris.to_numpy()

    ## Question 1 ##
    # My Implementations
    IrisLL = toArrayofLL(IrisMatrix)
    printAllLLinArray(IrisLL)
    remove20Percent(IrisLL)
    printAllLLinArray(IrisLL)
    print(len(IrisLL))
