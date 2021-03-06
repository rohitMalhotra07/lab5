%Input:
% classA -> matrix for class A
% classB -> matrix for class B
% npa -> number of prototypes for class A
% npb -> number of prototypes for class B
% lr -> learning rate
% num_epochs -> minimum number of epochs

%Output:
% PA -> Matrix of final prototypes of class A
% PB -> Matrix of final prototypes of class B.
% E -> contains error after each epoch
function [PA,PB,E] = train_lvq(classA,classB,npa,npb,lr,num_epochs)
shapeA = size(classA);
shapeB=size(classB);
% initialize prototypes
PA = initialize_for_one(classA,npa); 
PB = initialize_for_one(classB,npb);

%PA =  zeros(npa,shapeA(2));
%PB =  zeros(npb,shapeB(2));
E = zeros(1,1);
length_e = 1; % keep track of length of E, which will be number of epochs till now.
flag = true; %flag true means keep training
while flag
    %sample data for each epoch
    %classA_s = datasample(classA,randi([10 shapeA(1)],1));
    %classB_s = datasample(classB,randi([10 shapeB(1)],1));
    [PA,PB] = run_each_epoch(classA,classB,PA,PB,lr,true);
    e = calculate_error(classA,classB,PA,PB);
    last_e = E(length_e);
    E = [E ; e];
    length_e = length_e +1;
    %This is termination condition if diffeence of current and last one
    %error is zero and number of epochs is greater than minimum number of
    %epochs.
    %if abs(e - last_e) == 0 %&& length_e > num_epochs
        %flag =false;
    %end
    %This is termination condition when last three errors are same.
    if length_e >= 4 % for comparing last three error.
        last_three = E(size(E,1)-2:size(E,1),:); % last three elements of E
        if all(last_three == last_three(1))
            flag =false;
        end
    end
end

end

% following function is used to initialize the prototypes.
%function to initialize n number of prototypes of class as mean of n chunks of given samples.
function p = initialize_for_one(class,n)
shape = size(class);
class = class(randperm(shape(1)),:); %to get random samples
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

%disp(p);
end

%following function is used to calculate error
%inputs have usual meaning defined above
%outpur e is error.
function e = calculate_error(classA,classB,PA,PB)
shapeA=size(classA);
shapeB=size(classB);
rowsA = shapeA(1);% number of samples of class A
rowsB = shapeB(1); % number of samples of class B
total = rowsA+rowsB; % total number of samples

wrong = 0 ;%number of wrong classification

for i = [1:rowsA]
    da = min(pdist2(classA(i,:),PA,'euclidean').^2); %minimum distance of ith sample of class A with all PA
    db = min(pdist2(classA(i,:),PB,'euclidean').^2); %minimum distance of ith sample of class A with all PB
    
    if da>db
        wrong = wrong +1 ; % wrong classification
    end
end
for i = [1:rowsB]
    da = min(pdist2(classB(i,:),PA,'euclidean').^2); %minimum distance of ith sample of class B with all PA
    db = min(pdist2(classB(i,:),PB,'euclidean').^2); %minimum distance of ith sample of class B with all PB
    
    if db>da
        wrong = wrong +1 ; % wrong classification
    end
end

e = wrong/total;
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
        if present_sequentialy == true % if we want to present sequentialy.
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
    
    [da,xa] = min(pdist2(sample,PA,'euclidean').^2); %minimum distance of sample with all PA
    [db,xb] = min(pdist2(sample,PB,'euclidean').^2); %minimum distance of sample with all PB
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