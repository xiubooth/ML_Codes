function p = fw2(x)
%fw2 find the position of maximum of a matrix
maximum=max(max(x));
[X,Y]=find(x==maximum);
p=[X,Y];
end

