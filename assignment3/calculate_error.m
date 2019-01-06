%following function is used to calculate error
%inputs have usual meaning defined above
%l1 is lambda 1 and l2 is lambda 2.
%outpur e is error.
function e = calculate_error(classA,classB,PA,PB,l1,l2)
shapeA=size(classA);
shapeB=size(classB);
rowsA = shapeA(1);% number of samples of class A
rowsB = shapeB(1); % number of samples of class B
total = rowsA+rowsB; % total number of samples

wrong = 0 ;%number of wrong classification

for i = [1:rowsA]
    da = min(dist_global_relevance(l1,l2,classA(i,:),PA)); %minimum distance of ith sample of class A with all PA
    db = min(dist_global_relevance(l1,l2,classA(i,:),PB)); %minimum distance of ith sample of class A with all PB
    
    if da>db
        wrong = wrong +1 ; % wrong classification
    end
end
for i = [1:rowsB]
    da = min(dist_global_relevance(l1,l2,classB(i,:),PA)); %minimum distance of ith sample of class B with all PA
    db = min(dist_global_relevance(l1,l2,classB(i,:),PB)); %minimum distance of ith sample of class B with all PB
    
    if db>da
        wrong = wrong +1 ; % wrong classification
    end
end

e = wrong/total;
%disp(wrong);
end