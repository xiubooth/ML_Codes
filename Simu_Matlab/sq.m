function r = sq(a,b,step)
%sq 
%
r=[];
new=a;
r(end+1)=a;
for i=1:10000
    new=new+step;
    if new<=b
        r=r+[new];
    else 
        break
    end
end
end

