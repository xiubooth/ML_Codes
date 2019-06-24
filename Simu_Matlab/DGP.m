%%% This Version: July 30, 2018. @copyright Shihao Gu, Bryan Kelly and Dacheng Xiu
%%% If you use these codes, please cite the paper "Empirical Asset Pricing via Machine Learning" (2018) and "Autoencoder Asset Pricing Models." (2019)

%%% Generate Simulation Datasets %%%

for M=1:1
    path='./Simu'; % set your own folder path
    
    name1='/SimuData_p50'; % Case Pc=50 
    name2='/SimuData_p100'; % Case Pc=100
    mkdir(path);
    mkdir(sprintf('%s',path,name1));
    mkdir(sprintf('%s',path,name2));
    
    %%% Case Pc=100 %%%
    N=200;
    m=100;
    T=180;
    stdv=0.05;
    theta_w=0.02;
    stde=0.05;
    
    rho=unifrnd(0.9,1,[m,1]);
    c=zeros(N*T,m);
    for i=1:m
        x=zeros(N,T);
        x(:,1)=normrnd(0,1,[N,1]);
        for t=2:T
            x(:,t)=rho(i).*x(:,t-1)+normrnd(0,1,[N,1]).*sqrt(1-rho(i)^2);
        end
        [~,r]=sort(x);
        szx=size(x);
        x1=zeros(szx);
        ridx=1:szx(1);
        for k=1:szx(2)
            x1(r(:,k),k)=ridx*2/(N+1)-1;
        end
        c(:,i)=x1(:);
    end
    
    per=repmat(1:N,1,T);
    time=repelem(1:T,N);
    vt=normrnd(0,1,[3,T])*stdv;
    beta=c(:,[1,2,3]);
    betav=zeros(N*T,1);
    for t=1:T
        ind=(time==t);
        betav(ind)=beta(ind,:)*vt(:,t);
    end
        
    y=zeros(T,1);
    y(1)=normrnd(0,1);
    q=0.95;
    for t=2:T
        y(t)=q*y(t-1)+normrnd(0,1)*sqrt(1-q^2);
    end
        
    cy=c;
    for t=1:T
        ind=(time==t);
        cy(ind,:)=c(ind,:)*y(t);
    end

    ep=trnd(5,[N*T,1])*stde;
    
    
    %%% Model 1
    theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    r1=horzcat(c,cy)*theta'+betav+ep;
    rt=horzcat(c,cy)*theta';
    %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));
    
    pathc=sprintf('%s',path,name2);
    pathc=sprintf('%s',pathc,'/c');
    pathc=sprintf('%s%d',pathc,M);
    pathc=sprintf('%s',pathc,'.csv');
    csvwrite(pathc,horzcat(c,cy));
    
    pathr=sprintf('%s',path,name2);
    pathr=sprintf('%s',pathr,'/r1');
    pathr=sprintf('%s_%d',pathr,M);
    pathr=sprintf('%s',pathr,'.csv');
    csvwrite(pathr,r1);
    
    
    
    %%% Model 2
    theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    z=horzcat(c,cy);
    z(:,1)=c(:,1).^2*2;
    z(:,2)=c(:,1).*c(:,2)*1.5;
    z(:,m+3)=sign(cy(:,3))*0.6;
    
    r1=z*theta'+betav+ep;
    rt=z*theta';
    %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));
    
    pathr=sprintf('%s',path,name2);
    pathr=sprintf('%s',pathr,'/r2');
    pathr=sprintf('%s_%d',pathr,M);
    pathr=sprintf('%s',pathr,'.csv');
    csvwrite(pathr,r1);
    
    
    
    %%% Case Pc=50 %%%
    
    m=50;
    
    %%% MOdel 1
    
    theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    r1=horzcat(c(:,1:m),cy(:,1:m))*theta'+betav+ep;
    rt=horzcat(c(:,1:m),cy(:,1:m))*theta';
    %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));
    
    pathc=sprintf('%s',path,name1);
    pathc=sprintf('%s',pathc,'/c');
    pathc=sprintf('%s%d',pathc,M);
    pathc=sprintf('%s',pathc,'.csv');
    csvwrite(pathc,horzcat(c(:,1:m),cy(:,1:m)));
    
    pathr=sprintf('%s',path,name1);
    pathr=sprintf('%s',pathr,'/r1');
    pathr=sprintf('%s_%d',pathr,M);
    pathr=sprintf('%s',pathr,'.csv');
    csvwrite(pathr,r1);
    
   
    %%% Model 2
    
    theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    z=horzcat(c(:,1:m),cy(:,1:m));
    z(:,1)=c(:,1).^2*2;
    z(:,2)=c(:,1).*c(:,2)*1.5;
    z(:,m+3)=sign(cy(:,3))*0.6;
    
    r1=z*theta'+betav+ep;
    rt=z*theta';
    %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));
    
    pathr=sprintf('%s',path,name1);
    pathr=sprintf('%s',pathr,'/r2');
    pathr=sprintf('%s_%d',pathr,M);
    pathr=sprintf('%s',pathr,'.csv');
    csvwrite(pathr,r1);
    
    disp(M)

end