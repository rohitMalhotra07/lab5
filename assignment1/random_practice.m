clear all;
load('data_lvq_A.mat','matA');
load('data_lvq_B.mat','matB');

%training lvq1
[PA,PB,E] = train_lvq(matA,matB,2,1,0.01,25);
E_new = E(2:size(E,1));
figure;
X = [1:size(E_new,1)];
plot(X,E_new,'b','Marker','*');