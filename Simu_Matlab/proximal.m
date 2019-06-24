function v = proximal(groups,nc,XX,XY,tol,L,l2,func)
%proximal
%
dim=size(XX,1);
max_iter =30000;
gamma=1/L;
l1=l2;
% how to create float number
w=zeros(dim,1);
v=w;
for t=0:max_iter-1
    vold=v;
    w_prev=w;
    w=v-gamma*f_grad(XX,XY,v);
    w=func(groups,nc,w,l1*gamma);
    v=w+t/(t+3)*(w-w_prev);
    if (sum(power(v-vold,2)) < (sum(power(vold,2))*tol) || sum(abs(v-vold))==0)
        break
    end
end
end

