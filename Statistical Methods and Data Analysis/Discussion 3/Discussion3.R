# https://www.kaggle.com/ahsen1330/us-police-shootings

Shootings_data <- read.csv(file = "shootings.csv")
hist(Shootings_data$age)

mean(Shootings_data$age)
abline(v = mean(Shootings_data$age),
       col = "royalblue",
       lwd = 2)

median(Shootings_data$age)
abline(v = median(Shootings_data$age),
       col = "red",
       lwd = 2)

legend(x = "topright", # location of legend within plot area
       c("Mean", "Median"),
       col = c("royalblue", "red"),
       lwd = c(2, 2))

race = (Shootings_data$race)
M = table(race)
barplot(M, main="Race vs Freq",
        xlab="Race", ylab="Freq")



  