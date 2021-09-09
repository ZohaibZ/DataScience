import numpy as np
import matplotlib.pyplot as plt

def WorldSeriesSim(n,p):
    #n = Number of Series
    #p = Probability of A winning
    #Funtion will return [# of series won in 4 games, 5 games, 6 games, 7 games, percentage of series won out of n series]
    yserieswinpercentage = 0
    y4 = 0  #number of series won by A in 4 games
    y5 = 0  #number of series won by A in 5 games
    y6 = 0  #number of series won by A in 6 games   
    y7 = 0  #number of series won by A in 7 games
    ycurr = 0   #current series in while loop
    serieslength = 1    #current series length
    for i in range (1,n+1): #for the number of world series being evaluated
        while ycurr != 4 and serieslength <=7:  #checking for A winning series or series reaching length 7
            game = np.random.binomial(1, p, 1)  #user input of percentage
            if game == 1:
                ycurr += 1
            serieslength += 1

        serieslength -= 1 #while loop correction 
        if ycurr == 4 and serieslength == 4: #y wins in 4
            y4 += 1
        if ycurr == 4 and serieslength == 5: #y wins in 5
            y5 += 1
        if ycurr == 4 and serieslength == 6: #y wins in 6
            y6 += 1
        if ycurr == 4 and serieslength == 7: #y wins in 7
            y7 += 1
        
        serieslength = 1    #reset variables for next series
        ycurr = 0

    yserieswinpercentage = (y4+y5+y6+y7)/n
    return [y4, y5, y6, y7, yserieswinpercentage]

if __name__ == '__main__':
    Sim1 = WorldSeriesSim(1000, .55) # WorldSeriesSim(# of series, Probability of A winning)
    print(Sim1)