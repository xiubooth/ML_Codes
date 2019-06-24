function m=lossh(y,yhat,mu)
%lossh
%   
r=abs(yhat-y);
l=zeros(length(r),1);
ind=(r>mu);
l(ind)=2*mu.*r(ind)-mu.*mu;
ind=(r<=mu);
l(ind)=r(ind).*r(ind);
m=mean(l);
end

