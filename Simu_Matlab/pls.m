function B=pls(X,y,A)
%pls
%   
s=X.'*y;
R=zeros(size(X,2),A);
TT=zeros(size(X,1),A);
P=zeros(size(X,2),A);
U=zeros(size(X,1),A);
V=zeros(size(X,2),A);
B=zeros(size(X,2),A);
Q=zeros(1,A);

for i=0:(A-1)
    q=s.'*s;
    r=s*q;
    t=X*r;
    t=t-mean(t);
    normt=sqrt(t.'*t);
    t=t/normt;
    r=r/normt;
    p=X.'*t;
    q=y.'*t;
    u=y*q;
    v=p;
    if i>0
        v=v-V(:,1:(i))*(V(:,1:(i)).'*p);
        u=u-TT(:,1:(i))*(TT(:,1:(i)).'*u);
    end
    v=v/sqrt(v.'*v);
    s=s-v*(v.'*s);
    
    R(:,i+1)=r;
    TT(:,i+1)=t;
    P(:,i+1)=p;
    U(:,i+1)=u;
    V(:,i+1)=v;
    Q(:,i+1)=q;
end

for i=0:(A-2)
    C=R(:,1:(i+1))*Q(:,1:(i+1)).';
    B(:,i+2)=C(:,1);
end
end

