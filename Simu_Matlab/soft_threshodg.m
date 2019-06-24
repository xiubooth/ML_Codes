function w1=soft_threshodg(groups,nc,w,mu)
%soft_threshodg
%  
w1=w;
for i=1:nc
    ind=(groups==i);
    wg=w1(ind,:);
    nn=size(wg,1);
    n2=sqrt(sum(power(wg,2)));
    if n2<=mu
        w1(ind,:)=zeros(nn,1);
    else
        w1(ind,:)=wg-mu*wg/n2;
    end
end
end

