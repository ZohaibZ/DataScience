set.seed(10)
y = matrix(rbinom(7000,1,.55),1000,7, byrow = TRUE)
group = sample(1, 1000, TRUE)
RS = rowSums(y)
Wins = table(RS)

