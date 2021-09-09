import numpy as np
import pandas as pd
import math

# Below is a useful site that i used to understand a little bit about evaulating the board.
# https://www3.ntu.edu.sg/home/ehchua/programming/java/JavaGame_TicTacToe_AI.html

def evaluateBoard(board, xEval, oEval):
    #intitialization of variables
    xRowSum = [0]*3
    oRowSum = [0]*3
    xColSum = [0]*3
    oColSum = [0]*3
    oDiagSum1 = 0
    xDiagSum1 = 0
    oDiagSum2 = 0
    xDiagSum2 = 0

    xCount = 0
    oCount = 0
    for i in range(len(board)): # finding sums for the rows for both x and o
        for j in range(len(board)):
            if board[i][j] == x:
                xRowSum[i] += xEval[i][j]
                xCount += 1
            if board[i][j] == o:
                oRowSum[i] += oEval[i][j]
                oCount += 1
        xRowSum[i] *= xCount
        oRowSum[i] *= oCount
        xCount = 0
        oCount = 0

    xCount = 0
    oCount = 0
    for i in range(len(board)): # finding sums for the columns for both x and o
        for j in range(len(board)):
            if board[j][i] == x:
                xColSum[i] += xEval[j][i]
                xCount += 1
            if board[j][i] == o:
                oColSum[i] += oEval[j][i]
                oCount += 1
        xColSum[i] *= xCount
        oColSum[i] *= oCount
        xCount = 0
        oCount = 0

    xdiag1Count = 0
    odiag1Count = 0
    xdiag2Count = 0
    odiag2Count = 0

    for i in range(len(board)): # finding sums for the 2 diagonals both x and o
        for j in range(len(board)):
            if i == j:
                if board[i][j] == x:
                    xDiagSum1 += xEval[i][j]
                    xdiag1Count += 1
                if board[i][j] == o:
                    oDiagSum1 += oEval[i][j]
                    odiag1Count += 1
            if (i+j) == 2:
                if board[i][j] == x:
                    xDiagSum2 += xEval[i][j]
                    xdiag2Count += 1
                if board[i][j] == o:
                    oDiagSum2 += oEval[i][j]
                    odiag2Count += 1

    xDiagSum1 *= xdiag1Count
    oDiagSum1 *= odiag1Count
    xDiagSum2 *= xdiag2Count
    oDiagSum2 *= odiag2Count

    rowSub = sum(np.subtract(xRowSum,oRowSum))
    colSub = sum(np.subtract(xColSum,oColSum))
    diagSum1 = xDiagSum1 - oDiagSum1
    diagSum2 = xDiagSum2 - oDiagSum2
    Total = rowSub + colSub + diagSum1 + diagSum2
    return (Total)


 
    return
if __name__ == '__main__':
    o = 'o'
    x = 'x'
    Game1 = [[o,0,x],[0,o,0],[0,0,0]]
    Game2 = [[o,0,x],[0,0,0],[o,0,0]]
    Game3 = [[o,0,x],[0,0,0],[0,0,0]]
    Game4 = [[x,0,0],[0,o,o],[0,0,0]]
    Game5 = [[0,0,0],[x,o,0],[o,0,0]]
    Game6 = [[0,0,x],[0,o,x],[o,0,o]]
    Game7 = [[0,0,0],[0,o,0],[0,0,0]]
    Game8 = [[0,0,x],[0,0,0],[o,0,0]]
    Game9 = [[o,0,x],[0,0,0],[o,x,o]]
    Game10 = [[o,0,0],[x,x,o],[o,0,0]]
    Game11 = [[o,0,x],[o,0,0],[0,0,0]]
    Game12 = [[o,0,0],[0,0,0],[0,0,0]]
    Game13 = [[o,0,x],[0,0,0],[0,0,o]]
    Game14 = [[0,0,o],[0,0,0],[0,0,0]]
    oEval = [[8,4,7],[3,9,2],[6,1,5]]
    xEval = [[17,13,16],[12,18,11],[15,10,14]]

    G1 = evaluateBoard(Game1, xEval, oEval)
    print(Game1)
    print("G1:", G1)
    print("")
    
    G2 = evaluateBoard(Game2, xEval, oEval)
    print(Game2)
    print("G2:", G2)
    print("")
    
    G3 = evaluateBoard(Game3, xEval, oEval)
    print(Game3)
    print("G3:", G3)
    print("")
    
    G4 = evaluateBoard(Game4, xEval, oEval)
    print(Game4)
    print("G4:", G4)
    print("")
    
    G5 = evaluateBoard(Game5, xEval, oEval)
    print(Game5)
    print("G5:", G5)
    print("")
    
    G6 = evaluateBoard(Game6, xEval, oEval)
    print(Game6)
    print("G6:", G6)
    print("")
    
    G7 = evaluateBoard(Game7, xEval, oEval)
    print(Game7)
    print("G7:", G7)
    print("")
    
    G8 = evaluateBoard(Game8, xEval, oEval)
    print(Game8)
    print("G8:", G8)
    print("")
    
    G9 = evaluateBoard(Game9, xEval, oEval)
    print(Game9)
    print("G9:", G9)
    print("")
    
    G10 = evaluateBoard(Game10, xEval, oEval)
    print(Game10)
    print("G10:", G10)
    print("")

    G11 = evaluateBoard(Game11, xEval, oEval)
    print(Game11)
    print("G11:", G11)
    print("")

    G12 = evaluateBoard(Game12, xEval, oEval)
    print(Game12)
    print("G12:", G12)
    print("")

    G13 = evaluateBoard(Game13, xEval, oEval)
    print(Game13)
    print("G13:", G13)
    print("")

    G14 = evaluateBoard(Game14, xEval, oEval)
    print(Game14)
    print("G14:", G14)
    print("")