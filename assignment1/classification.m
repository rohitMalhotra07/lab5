%this function takes each sample and reclassifies it.
%input classA and classB are given matrices.
%input PA and PB are 
function [class_a,class_b]= classification(classA,classB,PA,PB)
class_a=[];
class_b=[];

for i = [1:size(classA,1)]
    da = min(pdist2(classA(i,:),PA,'euclidean').^2); %minimum distance of ith sample of class A with all PA
    db = min(pdist2(classA(i,:),PB,'euclidean').^2); %minimum distance of ith sample of class A with all PB
    
    if da<db
       class_a = [class_a;classA(i,:)];
    else
       class_b = [class_b;classA(i,:)];
    end
end

for i = [1:size(classB,1)]
    da = min(pdist2(classB(i,:),PA,'euclidean').^2); %minimum distance of ith sample of class A with all PA
    db = min(pdist2(classB(i,:),PB,'euclidean').^2); %minimum distance of ith sample of class A with all PB
    
    if da<db
       class_a = [class_a;classB(i,:)];
    else
       class_b = [class_b;classB(i,:)];
    end
end

end