%following function calculates global relevance distance of given sample
%with each weight in weights matrix and return pair wise distance of sample
%with each matrix
%l1 is lambda 1 and l2 is lambda 2.
function dist = dist_global_relevance(l1,l2,sample,weights)
rows = size(weights,1); % determine no. of weights.

%initialise dist matrix with zeroes.
dist=zeros(rows,1);
for i = [1:rows]
    weight = weights(i,:);
    d = ((weight(1,1)-sample(1,1))^2)*l1 + ((weight(1,2)-sample(1,2))^2)*l2;
    dist(i)=d;
end
%disp(dist);
end