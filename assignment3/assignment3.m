clear all;
load('data_lvq_A.mat','matA');
load('data_lvq_B.mat','matB');

%following figure is to plot dataset with final prototypes
figure;
scatter(matA(:,1),matA(:,2),'r','*'); %plotting class a with * sign
hold on
scatter(matB(:,1),matB(:,2),'b', '+'); % plotting class b with + sign

[PA,PB,E,L1,L2] = train_lvq(matA,matB,2,1,0.5,0.5,0.01,15);
hold on
scatter(PA(:,1),PA(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63]);
hold on
scatter(PB(:,1),PB(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.1 0.5 .63]);
title('S3801128');
xlabel('Feature1');
ylabel('Feature2');
legend({'ClassA','ClassB'});


%figure for Question 3
figure;
[PA,PB,E,L1,L2] = train_lvq(matA,matB,2,1,0.5,0.5,0.01,25);
subplot(1,2,1);
E_new = E(2:size(E,1));
X = [1:size(E_new,1)];
plot(X,E_new,'m');
hold on;
title('Training Error');
xlabel('Number of Epochs');
ylabel('Training Error');

subplot(1,2,2);
L1_new = L1(2:size(L1,1));
X1 = [1:size(L1_new,1)];
L2_new = L2(2:size(L2,1));
X2 = [1:size(L2_new,1)];
plot(X1,L1_new,'r');
hold on;
plot(X2,L2_new,'g');
legend({'Relevance1','Relevance2'});