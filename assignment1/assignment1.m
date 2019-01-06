clear all;
load('data_lvq_A.mat','matA');
load('data_lvq_B.mat','matB');

figure;
scatter(matA(:,1),matA(:,2),'r','*'); %plotting class a with * sign
hold on
scatter(matB(:,1),matB(:,2),'b', '+'); % plotting class b with + sign
title('S3801128');
xlabel('Feature1');
ylabel('Feature2');
legend({'ClassA','ClassB'});


%training lvq1
[PA,PB,E] = train_lvq(matA,matB,2,1,0.01,25);
E_new = E(2:size(E,1));
figure;
X = [1:size(E_new,1)];
plot(X,E_new,'b');

title('S3801128');
xlabel('Number of Epochs');
ylabel('Training Error');
ylim([0.1 inf]);
%legend({'ClassA','ClassB'});

figure;
%plot for different number of prototypes
%1-1
[PA,PB,E] = train_lvq(matA,matB,1,1,0.01,25);
E_new = E(2:size(E,1));
X = [1:size(E_new,1)];
plot(X,E_new,'b');
hold on;

%1-2
[PA,PB,E] = train_lvq(matA,matB,1,2,0.01,25);
E_new = E(2:size(E,1));
X = [1:size(E_new,1)];
plot(X,E_new,'r');
hold on;

%2-1
[PA,PB,E] = train_lvq(matA,matB,2,1,0.01,25);
E_new = E(2:size(E,1));
X = [1:size(E_new,1)];
plot(X,E_new,'g');
hold on;

%2-2
[PA,PB,E] = train_lvq(matA,matB,2,2,0.01,25);
E_new = E(2:size(E,1));
X = [1:size(E_new,1)];
plot(X,E_new,'m');
hold on;


title('S3801128');
xlabel('Number of Epochs');
ylabel('Training Error');
ylim([0.1 inf]);
legend({'1-1','1-2','2-1','2-2'});




%side by side subplots
figure;

[PA,PB,E] = train_lvq(matA,matB,1,1,0.01,10);
[class_a,class_b]=classification(matA,matB,PA,PB);
subplot(2,2,1);
scatter(class_a(:,1),class_a(:,2),'r','*'); %plotting class a with * sign
hold on
scatter(class_b(:,1),class_b(:,2),'b', '+'); % plotting class b with + sign
hold on
scatter(PA(:,1),PA(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63]);
hold on
scatter(PB(:,1),PB(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.1 0.5 .63]);
title('1-1');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'ClassA','ClassB'});


[PA,PB,E] = train_lvq(matA,matB,1,2,0.01,10);
[class_a,class_b]=classification(matA,matB,PA,PB);
subplot(2,2,2);
scatter(class_a(:,1),class_a(:,2),'r','*'); %plotting class a with * sign
hold on
scatter(class_b(:,1),class_b(:,2),'b', '+'); % plotting class b with + sign
hold on
scatter(PA(:,1),PA(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63]);
hold on
scatter(PB(:,1),PB(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.1 0.5 .63]);
title('1-2');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'ClassA','ClassB'});


[PA,PB,E] = train_lvq(matA,matB,2,1,0.01,10);
[class_a,class_b]=classification(matA,matB,PA,PB);
subplot(2,2,3);
scatter(class_a(:,1),class_a(:,2),'r','*'); %plotting class a with * sign
hold on
scatter(class_b(:,1),class_b(:,2),'b', '+'); % plotting class b with + sign
hold on
scatter(PA(:,1),PA(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63]);
hold on
scatter(PB(:,1),PB(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.1 0.5 .63]);
title('2-1');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'ClassA','ClassB'});


[PA,PB,E] = train_lvq(matA,matB,2,2,0.01,10);
[class_a,class_b]=classification(matA,matB,PA,PB);
subplot(2,2,4);
scatter(class_a(:,1),class_a(:,2),'r','*'); %plotting class a with * sign
hold on
scatter(class_b(:,1),class_b(:,2),'b', '+'); % plotting class b with + sign
hold on
scatter(PA(:,1),PA(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.49 1 .63]);
hold on
scatter(PB(:,1),PB(:,2),50,'MarkerEdgeColor','k','MarkerFaceColor',[.1 0.5 .63]);
title('2-2');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'ClassA','ClassB'});



