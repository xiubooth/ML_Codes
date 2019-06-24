function resultfinal = cut_knots_degree2(x,n,th)
%cut_knots_degree2
%   do we need to make copies of matrix
%   use for loop to make copies
[a,b] = size(x);
resultfinal = zeros(a,b*(n+1));
for i=1:b
    xcut = x(:,i);
    xcutnona=xcut;
    xcutnona(isnan(xcutnona))=0;
    index=((1-1*isnan(xcut))==1);
    
    t=th(:,i);
    
    x1=xcutnona;
    resultfinal(:,(n+1)*i-n)=x1-mean(x1);
    x1=power(xcutnona-t(1),2);
    resultfinal(:,(n+1)*i-n+1)=x1-mean(x1);
 
    for j=1:(n-1)
        x1=power(xcutnona-t(j+1),2).*(xcutnona>=t(j+1));
        resultfinal(:,(n+1)*i-n+1+j)=x1-mean(x1);
    end
end
end

