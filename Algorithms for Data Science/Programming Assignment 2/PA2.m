clear
clc
close all

weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
iris_data.X(1,:) = weka_data.sepallength.values;
iris_data.X(2,:) = weka_data.sepalwidth.values;
iris_data.X(3,:) = weka_data.petallength.values;
iris_data.X(4,:) = weka_data.petalwidth.values;
iris_data.y = [ones(1,50) ones(1,50).*2 ones(1,50).*3];

iris_class_1 = cov(iris_data.X(:,1:50)');
rnd_data = rand(4,150);
rnd_data = iris_class_1*rnd_data;

iris_data.X(5,:) = (rnd_data(1,:).*3)+1; 

X = (rnd_data(2,1:50))';
[l, n] = size(X);
Pmin = min(rnd_data(1,1:50));
Pmax = max(rnd_data(1,1:50));
a = 2; 
b = 3;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(1,1:50) = X';

X = (rnd_data(2,51:100))';
[l, n] = size(X);
Pmin = min(rnd_data(1,51:100));
Pmax = max(rnd_data(1,51:100));
a = 3; 
b = 4;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(1,51:100) = X';

X = (rnd_data(2,101:150))';
[l, n] = size(X);
Pmin = min(rnd_data(1,101:150));
Pmax = max(rnd_data(1,101:150));
a = 4; 
b = 5;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(1,101:150) = X';

iris_data.X(6,:) = iris_synthetic;