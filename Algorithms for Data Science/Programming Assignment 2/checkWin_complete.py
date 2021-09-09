# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:45:20 2019

@author: Samuel Fisher, Intern
Johns Hopkins University Applied Physics Laboratory
"""

#Display who won and add to win counter
def whoWin(x,End,Xwin,Owin): 
    Xwin = 0
    Owin = 0
    if x == 1:
        End.configure(text="Player 1 has won!", background = 'white')
        Xwin = 1
    elif x == 2:
        End.configure(text="Player 2 has won!", background = 'white')
        Owin = 1
    else:
        End.configure(text="Nobody Wins", background = 'white')
    gameover = 1
    L = [Xwin,Owin,gameover]
    return L

#Check if there is a three in a row
#If there is a win, a display which team one and count that win
def checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill): 
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        print ("Player",place[3]," wins")
        return whoWin(place[3],End,Xwin,Owin)
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        print ("Player",place[6]," wins")
        return whoWin(place[7],End,Xwin,Owin)
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return whoWin(3,End,Xwin,Owin)
        
    return [0,0,0]

# Check who won without calling whoWin
# Necessary for MiniMax
def checkWin2(place):

    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        return place[1]
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        return place[0]
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        return place[0]
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        return place[1]
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        return place[2]
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        return place[2]
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        return place[3]
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        return place[6]
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return 0
    
    return None

#Check possibilities for wins in the next move
def checkWinPos(place):
    #intitialization of variables

    # check for ai win positions first in all 8 possible win combinations
    firstcolumn_ai = (place[0] == place[1] == 2) or (place[0]==place[2] == 2) or (place[1] == place[2] == 2)
    secondcolumn_ai = (place[3] == place[4] == 2) or (place[3]==place[5] == 2) or (place[4] == place[5] == 2)
    thirdcolumn_ai = (place[6] == place[7] == 2) or (place[6]==place[8] == 2) or (place[7] == place[8] == 2)
    firstrow_ai = (place[0] == place[3] ==2) or (place[0]==place[6] ==2) or (place[3] == place[6] ==2)
    secondrow_ai = (place[1] == place[4] ==2) or (place[1]==place[7] ==2) or (place[4] == place[7] ==2)
    thirdrow_ai = (place[2] == place[5] ==2) or (place[2]==place[8] ==2) or (place[5] == place[8] ==2)
    diagonal1_ai= (place[0] == place[4] ==2) or (place[0]==place[8] ==2) or (place[4] == place[8] ==2)
    diagonal2_ai = (place[2] == place[4] ==2) or (place[2]==place[6] ==2) or (place[4] == place[6] ==2)

    # check for user win positions next in all 8 possible win combinations
    firstcolumn_user = (place[0] == place[1] == 1) or (place[0]==place[2] == 1) or (place[1] == place[2] == 1)
    secondcolumn_user = (place[3] == place[4] == 1) or (place[3]==place[5] == 1) or (place[4] == place[5] == 1)
    thirdcolumn_user = (place[6] == place[7] == 1) or (place[6]==place[8] == 1) or (place[7] == place[8] == 1)
    firstrow_user = (place[0] == place[3] ==1) or (place[0]==place[6] ==1) or (place[3] == place[6] ==1)
    secondrow_user = (place[1] == place[4] ==1) or (place[1]==place[7] ==1) or (place[4] == place[7] ==1)
    thirdrow_user = (place[2] == place[5] ==1) or (place[2]==place[8] ==1) or (place[5] == place[8] ==1)
    diagonal1_user= (place[0] == place[4] ==1) or (place[0]==place[8] ==1) or (place[4] == place[8] ==1)
    diagonal2_user = (place[2] == place[4] ==1) or (place[2]==place[6] ==1) or (place[4] == place[6] ==1)
    

    ai = firstcolumn_ai or secondcolumn_ai or thirdcolumn_ai or firstrow_ai or secondrow_ai or thirdrow_ai or diagonal1_ai or diagonal2_ai
    user = firstcolumn_user or secondcolumn_user or thirdcolumn_user or firstrow_user or secondrow_user or thirdrow_user or diagonal1_user or diagonal2_user

    if ai: # check if ai had a win position somewhere then place user
        if firstcolumn_ai: 
            if place[0] == 0:
                return 0
            if place[1] == 0:
                return 1
            if place[2] == 0:
                return 2
        if secondcolumn_ai: 
            if place[3] == 0:
                return 3
            if place[4] == 0:
                return 4
            if place[5] == 0:
                return 5
        if thirdcolumn_ai: 
            if place[6] == 0:
                return 6
            if place[7] == 0:
                return 7
            if place[8] == 0:
                return 8
        if firstrow_ai: 
            if place[0] == 0:
                return 0
            if place[3] == 0:
                return 3
            if place[6] == 0:
                return 6
        if secondrow_ai: 
            if place[1] == 0:
                return 1
            if place[4] == 0:
                return 4
            if place[7] == 0:
                return 7
        if thirdrow_ai: 
            if place[2] == 0:
                return 2
            if place[5] == 0:
                return 5
            if place[8] == 0:
                return 8
        if diagonal1_ai:
            if place[0] == 0:
                return 0
            if place[4] == 0:
                return 4
            if place[8] == 0:
                return 8
        if diagonal2_ai:
            if place[2] == 0:
                return 4
            if place[4] == 0:
                return 4
            if place[6] == 0:
                return 6
    elif user: # check if user had a win position somewhere then block it
        if firstcolumn_user: 
            if place[0] == 0:
                return 0
            if place[1] == 0:
                return 1
            if place[2] == 0:
                return 2
        if secondcolumn_user: 
            if place[3] == 0:
                return 3
            if place[4] == 0:
                return 4
            if place[5] == 0:
                return 5
        if thirdcolumn_user: 
            if place[6] == 0:
                return 6
            if place[7] == 0:
                return 7
            if place[8] == 0:
                return 8
        if firstrow_user: 
            if place[0] == 0:
                return 0
            if place[3] == 0:
                return 3
            if place[6] == 0:
                return 6
        if secondrow_user: 
            if place[1] == 0:
                return 1
            if place[4] == 0:
                return 4
            if place[7] == 0:
                return 7
        if thirdrow_user: 
            if place[2] == 0:
                return 2
            if place[5] == 0:
                return 5
            if place[8] == 0:
                return 8
        if diagonal1_user:
            if place[0] == 0:
                return 0
            if place[4] == 0:
                return 4
            if place[8] == 0:
                return 8
        if diagonal2_user:
            if place[2] == 0:
                return 4
            if place[4] == 0:
                return 4
            if place[6] == 0:
                return 6

    return None