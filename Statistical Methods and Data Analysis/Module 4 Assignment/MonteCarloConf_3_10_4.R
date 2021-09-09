Fy <- function(y) {y^2}
X <- function (U) {sqrt(U)}

# Desired number of samples:
N = 100000

# position counter
i = 0

# Contains .6 Counter 
x = 0

while( i < N )
{
  U = runif(5) 
  y = X(U)
  y = sort(y)
  if ((y[1]<.6) & (y[5] > .6)){
    x = x+1
  } 
  i = i+1
}

Probability = x  / N

