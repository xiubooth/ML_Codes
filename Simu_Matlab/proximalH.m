
function a=proximalH(groups,nc,xtest,mtrain,ytest,w,X,y,mu,tol,L,l2,func)
%proximalH
%   
dim=size(X,1);
max_iter=3000;
gamma=1/L;
l1=l2;
v=w;
yhatbig1=xtest*w+mtrain;
r20=lossh(yhatbig1,ytest,mu);
for t=0:max_iter-1
    vold=v;
    w_perv=w;
    w=v-gamma*f_gradh(v,X,y,mu);
    mu1=l1*gamma;
    w=func(groups,nc,w,mu1);
    v=w+t/(t+3)*(w-w_perv);
    if (sum(power(v-vold,2)) < (sum(power(vold,2))*tol) || sum(abs(v-vold))==0)
        break
    end
    %yhatbig1=xtest*v+mtrain;
    %r2=lossh(yhatbig1,ytest,mu);
    %if r2<r20
    %    r20=r2;
    %else
    %    break
    %end
end
a=v;
end

