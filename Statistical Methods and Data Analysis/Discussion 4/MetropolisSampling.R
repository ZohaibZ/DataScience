funct <- function(x) {dnorm(x,-7,2)*1/3 + dnorm(x,-2,3)*1/3 + dnorm(x,3,1) *1/3}

# Desired number of samples:
N = 100

# Container for sampled data:
samples = c()

# Initial position:
x = 0

# Size of maximum jump:
step = 1

trials = 0

# Sample until enough samples are collected:

while( length(samples) < N )
{
  trials = trials + 1
  
  # Propose a new location:
  xp = x + runif(1,-1,1) * step
  
  # Measure likelihood ratio:
  r = funct(xp)/funct(x)
  
  # Accept if new point has higher probability density:
  if( r >= 1 )
  {
    samples = append(samples, xp)
    x = xp
  }
  # Accept with lower probability if new point has lower probability density:
  else
  {
    s = runif(1,0,1)
    if( s <= r )
    {
      samples=append(samples, xp)
      x = xp
    }
  }
}

cat( "Acceptance Rate =", N/trials )
hist(samples, breaks=30, probability = TRUE ,col=gray(.9), lwd=2, ylim=c(0,0.2))
curve(funct(x), add=T, lwd=2)
