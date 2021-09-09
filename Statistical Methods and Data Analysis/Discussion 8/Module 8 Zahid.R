library(dplyr)
library(ggplot2)
BorderData <-Border_Crossing_Entry_Data
BorderData$`Port Code`<-as.double(BorderData$`Port Code`)
BorderData$Date<-as.Date(BorderData$Date,format='%m/%d/%Y')
BorderData

A <- aggregate(BorderData$Value, by=list(Border = BorderData$Border, Time = BorderData$Date), FUN=sum)

TableA <- ggplot(A,aes(x=Time,y=x))+
  geom_line(aes(group=Border,color=Border))+
  theme_classic()+
  xlab("Time")+
  ylab("Total Traffic (pedestrians, vehicles, trains, etc.)")+
  ggtitle("Total Inbound Traffic to US over the 2 borders")

TableA

B <- aggregate(BorderData$Value, by=list(Travel = BorderData$Measure, Border = BorderData$Border, Time = BorderData$Date), FUN=sum)
B_Canada <- B[B$Border == 'US-Canada Border',]
B_Mexico <- B[B$Border == 'US-Mexico Border',]

TableB <- ggplot(B,aes(x=Time,y=x))+
  geom_line(aes(group=Travel,color=Travel))+
  theme_classic()+
  xlab("Time")+
  ylab("Total Traffic (pedestrians, vehicles, trains, etc.)")+
  ggtitle("Total Inbound Traffic by Vehicle type to US")

TableB

TableB_Canada <- ggplot(B_Canada,aes(x=Time,y=B_Canada$x))+
  geom_line(aes(group=Travel,color=Travel))+
  theme_classic()+
  xlab("Time")+
  ylab("Total Traffic (pedestrians, vehicles, trains, etc.)")+
  ggtitle("Total Inbound Traffic to US over Canada Border")

TableB_Canada


TableB_Mexico <- ggplot(B_Mexico,aes(x=Time,y=B_Mexico$x))+
  geom_line(aes(group=Travel,color=Travel))+
  theme_classic()+
  xlab("Time")+
  ylab("Total Traffic (pedestrians, vehicles, trains, etc.)")+
  ggtitle("Total Inbound Traffic to US over over Mexico Border")

TableB_Mexico

