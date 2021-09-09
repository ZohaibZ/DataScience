% This code is for educational and research purposes of comparisons. This
% is a numerical reader and processing for generating features. The end
% result will be a 10 class number data set where each observation will
% have 12 features from the 28 x 28 pixel size images. 
%
% References:
%    https://www.kaggle.com/c/digit-recognizer/data

clear;
clc;
close all;

train = readmatrix('train.csv');
(column, row)
classLabels = train(:,1);

features = [];
for i = 1:1000
    img = reshape(train(i,2:end),[28,28])';
    imgDCT = dct2(img);
    diagDCT = diag(imgDCT); % This contains the diagonal values of the image.
    rowDCT = imgDCT(:,1); % This contains the horizontal values of the image.
    colDCT = imgDCT(1,:); % This contains the vertical vallues of the image.
    features(i,:) = ...
        [mean(diagDCT) std(diagDCT) skewness(diagDCT) kurtosis(diagDCT) ...
         mean(rowDCT) std(rowDCT) skewness(rowDCT) kurtosis(rowDCT) ...
         mean(colDCT) std(colDCT) skewness(colDCT) kurtosis(colDCT) ...
         classLabels(i)];
end

numbrs = [2, 1, 17, 8, 4, 9, 22, 7, 11, 12];
figure,
subplot(3,4,1),imagesc(reshape(train(2,2:end),[28,28])'), colormap(gray)
subplot(3,4,2),imagesc(reshape(train(1,2:end),[28,28])'), colormap(gray)
subplot(3,4,3),imagesc(reshape(train(17,2:end),[28,28])'), colormap(gray)
subplot(3,4,4),imagesc(reshape(train(8,2:end),[28,28])'), colormap(gray)
subplot(3,4,5),imagesc(reshape(train(4,2:end),[28,28])'), colormap(gray)
subplot(3,4,6),imagesc(reshape(train(9,2:end),[28,28])'), colormap(gray)
subplot(3,4,7),imagesc(reshape(train(22,2:end),[28,28])'), colormap(gray)
subplot(3,4,8),imagesc(reshape(train(7,2:end),[28,28])'), colormap(gray)
subplot(3,4,9),imagesc(reshape(train(11,2:end),[28,28])'), colormap(gray)
subplot(3,4,10),imagesc(reshape(train(12,2:end),[28,28])'), colormap(gray)

writematrix(features,'trainFeatures.xls');