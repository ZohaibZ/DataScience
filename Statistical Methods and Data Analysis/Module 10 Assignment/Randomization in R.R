reps <- 10000
results <- numeric(reps)
X = c(3.525, 3.625, 3.383, 3.625, 3.661, 3.791, 3.941, 3.781, 3.660, 3.733)
muX = mean(X)
Y = c(2.923, 3.385, 3.154, 3.363, 3.226, 3.283, 3.427, 3.437, 3.746, 3.438)
muY = mean(Y)
diff = muX-muY
x <- c(X,Y)

for(i in 1:reps){
  temp <- sample(x)
  results[i]<- mean(temp[1:10])-mean(temp[11:20])
}

abs_results = abs(results)
hist(results)
p.value <- sum(abs_results >= diff) / reps
t.test(X,Y, alternate="greater", lower.tail=FALSE)
