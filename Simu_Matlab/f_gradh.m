function grad = f_gradh(w,X,y,mu)
%f_gradh
%   
r = X*w-y;
ind0 = find(abs(r)<=mu);
ind1 = find(r>mu);
indf1 = find(r<-mu);
grad = X(ind0,:).'*(X(ind0,:)*w-y(ind0))+mu*X(ind1,:).'*ones(length(ind1),1)-mu*X(indf1,:).'*ones(length(indf1),1);
end
