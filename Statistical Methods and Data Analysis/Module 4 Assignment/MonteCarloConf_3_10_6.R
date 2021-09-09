Fy <- function(y) {1 - exp(-y)}
X <- function(U){-log(1-U)}

# Desired number of samples:
N = 1000

# position counter
i = 0

# Contains .6 Counter 
x = 0

while( i < N )
{
  U = runif(12) 
  y = X(U)
  y = sort(y)
  if ((y[1]<.2)){
    x = x+1
  } 
  i = i+1
}

Probability = x  / N

