%Input:
% classA -> matrix for class A
% classB -> matrix for class B
% npa -> number of prototypes for class A
% npb -> number of prototypes for class B
% lr -> learning rate
% num_epochs -> number of epochs

%Output:
% PA -> Matrix of final prototypes of class A
% PB -> Matrix of final prototypes of class B.
function [PA,PB] = train_lvq(classA,classB,npa,npb,lr,num_epochs)
shapeA = size(classA);
shapeB=size(classB);
% initialize prototypes
PA = initialize_for_one(classA,npa); 
PB = initialize_for_one(classB,npb);


length_e = 0; % number of epochs till now.
flag = true; %flag true means keep training
while flag
    
    [PA,PB] = run_each_epoch(classA,classB,PA,PB,lr,true);
    length_e = length_e +1;
    if length_e >= num_epochs
        flag =false;
    end
end

end

% following function is used to initialize the prototypes.
%function to initialize n number of prototypes of class as mean of n chunks of given samples.
function p = initialize_for_one(class,n)
shape = size(class);
class = class(randperm(shape(1)),:);%to draw random samples
p =  zeros(n,shape(2)); %initialize prototypes of class as all zeroes

start_i=1; %initialization of starting index of chunk
step = floor(shape(1)/n);
l=0; % initializing l as zero for case when n = 1
for i = [1:(n-1)]
    p(i,:) = mean(class(start_i:start_i+step-1,:));
    start_i = start_i + step;
    l=i;
end
p(l+1,:) = mean(class(start_i:shape(1),:)); % last prototype will be the mean of last remaining items

end

%following function is code for each epoch
%input is classA and classB samples 
%PA and PB are initialised weights or final weights after the last epoch
%lr is learning rate
%present_sequentialy if true presents samples sequentialy if false presents
%sample randomly
%output is pa pb after the current epoch
function [PA,PB] = run_each_epoch(classA,classB,PA,PB,lr,present_sequentialy)
shapeA=size(classA);
shapeB=size(classB);
rowsA = shapeA(1);% number of samples of class A
rowsB = shapeB(1); % number of samples of class B
total = rowsA+rowsB; % total number of samples

a_i = 1; % initialization of index of current element in classA
b_i = 1; % initialization of index of current element in classB

for i = [1:total]
    %for presenting samples randomly from each class during training
    %ran_val is used
    %rand_val 0 means select next from class A and 1 means select next of class B
    if a_i > rowsA %means all elements of classA are seen
        rand_val = 1;
    elseif b_i > rowsB % means all elements of classb are seen
        rand_val = 0;
    else
        if present_sequentialy
            rand_val =0; %first present A
        else
            rand_val = randsample([0 1],1);
        end
        
    end
    
    if rand_val == 0
        sample = classA(a_i,:);
        a_i = a_i+1;
    else
        sample = classB(b_i,:);
        b_i = b_i+1;
    end
    
    %now we have point as sample and its class as rand_val (0 -> A, 1->B)
    
    [da,xa] = min(pdist2(sample,PA,'euclidean')); %minimum distance of sample with all PA
    [db,xb] = min(pdist2(sample,PB,'euclidean')); %minimum distance of sample with all PB
    if da<db %means sample is calculated as class a
        if rand_val == 0 %means calculated and given classes are same so pull
            pull = 1;
        else
            pull = -1; % repel
        end
    else
        if rand_val == 0 %means calculated and given classes are different so repel
            pull = -1;
        else
            pull = 1; % pull
        end
    end
    
    %now update weights
    if da<db
        PA(xa,:) = PA(xa,:) + (lr * pull * (sample - PA(xa,:)));
    else
        PB(xb,:) = PB(xb,:) + (lr * pull * (sample - PB(xb,:)));
    end
    
end
end