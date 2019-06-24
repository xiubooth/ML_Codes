function val=soft_threshoda(w,alpha,mu)
%soft_threshoda
%   
val=sign(w).*max(abs(w)-alpha*mu,0)/(1+alpha*mu);
end

