function p = fw1(x)
%fw1 find the maximum location of a vector
%   
maximum=max(x);
p=find(x==maximum);
if length(p)>1
    p=p(1);
end
end

