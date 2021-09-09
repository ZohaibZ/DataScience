% This is a two and three class example using the iris data set.

clear;
clc;
close all;

load('irisNormalized.mat');
Data.X = X';
Data.Y = y';
model =  fishersMultiClassFeatureRanking(Data,1)

weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
iris_data.X(:,1) = weka_data.sepallength.values';
iris_data.X(:,2) = weka_data.sepalwidth.values';
iris_data.X(:,3) = weka_data.petallength.values';
iris_data.X(:,4) = weka_data.petalwidth.values';
iris_data.Y = [ones(1,50) ones(1,50).*2 ones(1,50).*3]';
model =  fishersMultiClassFeatureRanking(iris_data,1)

Data.X = X(:,1:100)';
Data.Y = [ones(1,50) ones(1,50).*-1]';
model =  fishersFeatureRanking(Data)
model =  bhattacharyyaFeatureRanking(Data)

iris_Data.X = iris_data.X(1:100,:);
iris_Data.Y = [ones(1,50) ones(1,50).*-1]';
model =  fishersFeatureRanking(iris_Data)
model =  bhattacharyyaFeatureRanking(iris_Data)