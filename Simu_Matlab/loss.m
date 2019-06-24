function m = loss(y,yhat)
%mean
% 
m=mean(power(yhat-y,2));
end

