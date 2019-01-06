%Input:
% classA -> matrix for class A
% classB -> matrix for class B
% npa -> number of prototypes for class A
% npb -> number of prototypes for class B
%l1 is lambda 1 and l2 is lambda 2.
% lr -> learning rate
% num_epochs -> minimum number of epochs

%Output:
% PA -> Matrix of final prototypes of class A
% PB -> Matrix of final prototypes of class B.
% E -> contains error after each epoch
% L1 -> contains values of relevance 1 after each epoch
% L2 -> contains values of relevance 2 after each epoch
function [PA,PB,E,L1,L2] = train_lvq(classA,classB,npa,npb,l1,l2,lr,num_epochs)
shapeA = size(classA);
shapeB=size(classB);
% initialize prototypes
PA = initialize_for_one(classA,npa); 
PB = initialize_for_one(classB,npb);

%PA =  zeros(npa,shapeA(2));
%PB =  zeros(npb,shapeB(2));
E = zeros(1,1);
L1 = zeros(1,1);
L2 = zeros(1,1);
length_e = 1; % keep track of length of E, which will be number of epochs till now.
flag = true; %flag true means keep training
while flag
    %sample data for each epoch
    %classA_s = datasample(classA,randi([10 shapeA(1)],1));
    %classB_s = datasample(classB,randi([10 shapeB(1)],1));
    [PA,PB,l1,l2] = run_each_epoch(classA,classB,PA,PB,l1,l2,lr,true);
    e = calculate_error(classA,classB,PA,PB,l1,l2);
    last_e = E(length_e);
    E = [E ; e];
    L1= [L1;l1];
    L2=[L2;l2];
    length_e = length_e +1;
    if abs(e - last_e) < 0.0001 && length_e > num_epochs
        flag =false;
    end
end

end

% following function is used to initialize the prototypes.
%function to initialize n number of prototypes of class as mean of n chunks of given samples.
function p = initialize_for_one(class,n)
shape = size(class);
class = class(randperm(shape(1)),:); %to get random samples (shuffling)
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
%l1 is lambda 1 and l2 is lambda 2.
%lr is learning rate
%present_sequentialy if true presents samples sequentialy if false presents
%sample randomly
%output is pa pb after the current epoch
function [PA,PB,l1,l2] = run_each_epoch(classA,classB,PA,PB,l1,l2,lr,present_sequentialy)
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
    
    [da,xa] = min(dist_global_relevance(l1,l2,sample,PA)); %minimum distance of sample with all PA
    [db,xb] = min(dist_global_relevance(l1,l2,sample,PB)); %minimum distance of sample with all PB
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
        [l1,l2]=updateLamda(l1,l2,sample,PA(xa,:),lr,pull);
    else
        PB(xb,:) = PB(xb,:) + (lr * pull * (sample - PB(xb,:)));
        [l1,l2]=updateLamda(l1,l2,sample,PB(xb,:),lr,pull);
    end
    
end
%disp(l1+l2);
end

%following function takes old values of lambda ond returns new values of
%lambda according to the update rule.
function [l1u,l2u]=updateLamda(l1,l2,sample,weight,lr,pull)
l1u = l1 - (lr*pull*(abs(sample(1,1)-weight(1,1))));
l2u = 1 - l1u ; % to enforce there sum is one.
if (l1u < 0) || (l2u < 0) % return old values in this case.
    l1u = l1;
    l2u = l2;
end
end