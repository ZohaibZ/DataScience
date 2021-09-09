# Iris Dataset

library(gplots)

par(mar=c(7,4,4,2)+0.1) 
png(filename='test.png', width=800, height=750)
df<-read.csv("iris.csv", header = TRUE)
Iris_matrix<-data.matrix(df[,1:4])
cormat<-signif(cor(Iris_matrix),2)
Iris_corMat<- heatmap(cormat,
                      margins=c(15,10),
                      col = heat.colors(256))

Iris_HeatMap <- heatmap(Iris_matrix, 
                      col = heat.colors(256),
                      Colv = NA, 
                      Rowv = NA,
                      scale="column")

dev.off()
