clear all;
load('data_lvq_A.mat','matA');
load('data_lvq_B.mat','matB');
k=10; %value for k in kfold cross validation

chunksA=createChunks(matA,k,true);
chunksB=createChunks(matB,k,true);

bar_data = zeros(k,1); %initializing test error as 0.

for i = [1:k]
    train_a = [];
    test_a = chunksA{i};
    train_b = [];
    test_b = chunksB{i};
    
    for l = [1:k]
        if l ~= i %other than ith chunk merge all chunks to form train set
            train_a = [train_a;chunksA{i}];
            train_b = [train_b;chunksB{i}];
        end
    end
    
    %now we have training data for class A and B and testing data for class
    %a and b
    
    [PA,PB,E,L1,L2] = train_lvq(matA,matB,2,1,0.5,0.5,0.01,15);
    lL1 = size(L1,1);%length of L1
    lL2 = size(L1,1);%length of L2
    e = calculate_error(test_a,test_b,PA,PB,L1(lL1,1),L2(lL2,1));
    bar_data(i) = e;
end

%plot bar graph
mean_error = mean(bar_data);
bar_data = bar_data*100;
mean_error = mean_error*100;
figure;


X=[1:k];
bar(X,bar_data,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5);
hold on;
yline(mean_error,'-.b');

title('s3801128');
ylabel('Error%');
xlabel('Test Fold');

 

% following function is used to create n chunks.
%Input: classGiven is the matrix or sample data given of one class
%p is output of n chunks.
%if random is true then make chunks randomly.
function p = createChunks(classGiven,n,random)
class = classGiven;

shape = size(class);
if random == true
    class = class(randperm(shape(1)),:);
end
%p=zeros(shape(1),shape(2),n);
p={};

start_i=1; %initialization of starting index of chunk
step = floor(shape(1)/n);
l=0; % initializing l as zero for case when n = 1
for i = [1:(n-1)]
    %p(:,:,i) = class(start_i:start_i+step-1,:);
    p{i} = class(start_i:start_i+step-1,:);
    start_i = start_i + step;
    l=i;
end
% last chunk will be the last remaining items
%p(:,:,l+1) = class(start_i:shape(1),:);
p{l+1} = class(start_i:shape(1),:);

end