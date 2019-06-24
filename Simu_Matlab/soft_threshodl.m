function val = soft_threshodl(groups,nc,w,mu)
%soft_threshodl
%   
val=sign(w).*max(abs(w)-mu,0);
end

