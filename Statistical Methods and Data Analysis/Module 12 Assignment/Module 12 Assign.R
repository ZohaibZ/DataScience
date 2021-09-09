# (r,s) = (1,1) & (4,4)
# (n,k) = (4,3) & (20,11)

prior <-function(r,s,theta){ 
  (1/beta(r,s))*(theta^(r-1))*((1-theta)^(s-1))
  }
posterior <-function(r,s,n,k,theta){
  (1/beta((r+k),(s+n-k)))*(theta^(r+k-1))*((1-theta)^(n+s-k-1))
}

theta <- seq(0,1,by=.01)

## 1 ##
# r= 1, s=1, n= 4, k=3
r=1
s=1
n=4
k=3

p_prior <- prior(r,s,theta)
p_posterior <- posterior(r,s,n,k,theta)
plot(theta,p_posterior, type="l",ylab="Probability")
lines(theta,p_prior)

a = r+k
b = s+n-k
p = 1 - pbeta(.5,a,b)
p

## 2 ##
# r= 4, s=4, n= 4, k=3
r=4
s=4
n=4
k=3

p_prior <- prior(r,s,theta)
p_posterior <- posterior(r,s,n,k,theta)
plot(theta,p_posterior, type="l",ylab="Probability")
lines(theta,p_prior)

a = r+k
b = s+n-k
p = 1 - pbeta(.5,a,b)
p

## 3 ##
# r= 1, s=1, n= 20, k=11
r=1
s=1
n=20
k=11

p_prior <- prior(r,s,theta)
p_posterior <- posterior(r,s,n,k,theta)
plot(theta,p_posterior, type="l",ylab="Probability")
lines(theta,p_prior)

a = r+k
b = s+n-k
p = 1 - pbeta(.5,a,b)
p

## 4 ##
# r= 4, s=4, n= 20, k=11
r=4
s=4
n=20
k=11

p_prior <- prior(r,s,theta)
p_posterior <- posterior(r,s,n,k,theta)
plot(theta,p_posterior, type="l", ylab="Probability")
lines(theta,p_prior)

a = r+k
b = s+n-k
p = 1 - pbeta(.5,a,b)
p

