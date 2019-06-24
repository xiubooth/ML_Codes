%%% This Version: July 30, 2018. @copyright Shihao Gu, Bryan Kelly and Dacheng Xiu
%%% If you use these codes, please cite the paper "Empirical Asset Pricing via Machine Learning" (2018) and "Autoencoder Asset Pricing Models." (2019)

%%% All regression models %%%

MC=1; % setup MC number

datanum=50; %Or datanum=100; seperately run two cases
path='./Simu'; % set your own folder path    
dirstock=strcat(path,'/SimuData_p',int2str(datanum),'/');




for hh=[1]
%for hh = [1 3 6 12] % correspond to monthly quarterly halfyear and annually returns
    
    title=strcat(path,'/Simu_p',int2str(datanum),'/Reg',int2str(hh));

    if (~(exist(title,'dir')==7) && MC==1)
        mkdir(title);
    end
    titleB = sprintf('%s',title,'/B');
    if (~(exist(titleB,'dir')==7) && MC==1)
        mkdir(titleB);
    end
    if datanum ==50
        nump=50;
    end
    if datanum ==100
        nump=100;
    end
    
    mu=0.2*sqrt(hh);
    tol=1e-10;
    
    
    % Start to MCMC
    for M=[MC]
        for mo=[1,2]
            
            disp(strcat('### MCMC :',int2str(M),', Model :',int2str(mo),' ###'))
            N=200;    % Number of CS tickers
            m=nump*2;    % Number of Characteristics
            T=180;    % Number of Time Periods
            
            per=repmat(1:N,1,T);
            time=repelem(1:T,N);
            stdv=0.05;
            theta_w=0.005;
            
            %%% Read Files
            path1=strcat(dirstock,'c',int2str(M),'.csv');
            path2=strcat(dirstock,'r',int2str(mo),'_',int2str(M),'.csv');
            c=csvread(path1,0,0);
            r1=csvread(path2,0,0);
            
            %%% Add Some Elements %%%

            daylen=repelem(N,T/3);
            daylen_test=daylen;
            ind=1:floor(N*T/3);
            xtrain=c(ind,:);
            ytrain=r1(ind);
            trainper=per(ind);
            ind=floor(N*T/3)+1:floor(N*(T*2/3-hh+1));
            xtest=c(ind,:);
            ytest=r1(ind);
            testper=per(ind);
            
            l1=size(c,1);
            l2=length(r1);
            l3=l2-sum(isnan(r1));
            
            ind=floor(N*T*2/3)+1:min([l1 l2 l3]);
            xoos=c(ind,:);
            yoos=r1(ind);
            clearvars c r1
            
            %%% Monthly Demean %%%
            ytrain_demean=ytrain-mean(ytrain);
            ytest_demean=ytest-mean(ytest);
            mtrain=mean(ytrain);
            mtest=mean(ytest);
            
            %%% Calcaulate Sufficient Stats %%%
            sd=zeros(size(xtrain,2),1); % dim of sd?
            for i=1:size(xtrain,2)
                s=std(xtrain(:,i));
                if s>0
                    xtrain(:,i)=xtrain(:,i)/s;
                    xtest(:,i)=xtest(:,i)/s;
                    xoos(:,i)=xoos(:,i)/s;
                    sd(i)=s;
                end
            end
            
            XX=xtrain.'*xtrain;
            [U,S,V]=svd(XX);
            L=S(1);
            %disp 'Lasso L = '
            %disp(L)
            Y=ytrain_demean;
            XY=xtrain.'*Y;
            
            %%% Start to Train %%%
            
            %%% OLS %%%
            r2_oos=zeros(13,1);  %%% OOS R2
            r2_is=zeros(13,1);  %%% IS R2
            
            modeln=1;
            groups=0;nc=0;
            clf=fitlm(xtrain,ytrain_demean,'Intercept',false);
            yhatbig1=predict(clf,xoos)+mtrain;
            r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=predict(clf,xtrain)+mtrain;
            r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            b=clf.Coefficients.Estimate;
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Simple OLS R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            func=@soft_threshodl;
            b=proximalH(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,0,func);
            yhatbig1=xoos*b+mtrain;
            r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=xtrain*b+mtrain;
            r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Simple OLS+H R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            ne=30;
            X=xtrain.'*xtrain;
            [pca_vec,pca_val]=eig(X);
            p1=pca_vec(:,size(pca_vec,2):-1:(size(pca_vec,2)-ne+1));
            Z=xtrain*p1;
            
            r=zeros(3,ne);
            B=zeros(size(xtrain,2),ne);
            Y=ytrain_demean;
            
            for j=1:ne-1
                xx=Z(:,1:j);
                b=(inv(xx.'*xx)*xx.')*Y;
                b=p1(:,1:j)*b;
                
                yhatbig1=xtest*b+mtrain;
                r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                yhatbig1=xoos*b+mtrain;
                r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=xtrain*b+mtrain;
                r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
                B(:,j)=b;
            end
            b=zeros(size(xtest,2),1);
            j=ne;
            yhatbig1=xtest*b+mtrain;
            r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
            yhatbig1=xoos*b+mtrain;
            r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=xtrain*b+mtrain;
            r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            B(:,j)=b;
            
            r2_oos(modeln)=r(2,int16(fw1(r(1,:))));
            r2_is(modeln)=r(3,int16(fw1(r(1,:))));
            b=B(:,int16(fw1(r(1,:))));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('PCR R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            B=pls(xtrain,ytrain_demean,30);
            ne=30;
            r=zeros(3,ne);
            Y=ytrain_demean;
            
            for j=1:ne
                b=B(:,j);
                yhatbig1=xtest*b+mtrain;
                r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                yhatbig1=xoos*b+mtrain;
                r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=xtrain*b+mtrain;
                r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            end
            
            r2_oos(modeln)=r(2,int16(fw1(r(1,:))));
            r2_is(modeln)=r(3,int16(fw1(r(1,:))));
            b=B(:,int16(fw1(r(1,:))));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('PLS R2 : ',num2str(r2_oos(modeln),3)));
            
            
            %%% Lasso %%%
            modeln=modeln+1;
            lamv=-2:0.1:4;
            alpha=1;
            r=zeros(3,length(lamv));
            
            for j=1:length(lamv)
                l2=10^lamv(j);
                func=@soft_threshodl;
                b=proximal(groups,nc,XX,XY,tol,L,l2,func);
                yhatbig1=xtest*b+mtrain;
                r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                yhatbig1=xoos*b+mtrain;
                r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=xtrain*b+mtrain;
                r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            end
            
            r2_oos(modeln)=r(2,int16(fw1(r(1,:))));
            r2_is(modeln)=r(3,int16(fw1(r(1,:))));
            l2=10^lamv(int16(fw1(r(1,:))));
            
            func=@soft_threshodl;
            b=proximal(groups,nc,XX,XY,tol,L,l2,func);
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Lasso R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            func=@soft_threshodl;
            b=proximalH(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,func);
            yhatbig1=xoos*b+mtrain;
            r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=xtrain*b+mtrain;
            r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Lasso+H R2 : ',num2str(r2_oos(modeln),3)));
            
            
            
            %%% Ridge %%%
            modeln=modeln+1;
            lamv=0:0.1:6;
            alpha=1;
            r=zeros(3,length(lamv));
            
            for j=1:length(lamv)
                l2=10^lamv(j);
                func=@soft_threshodr;
                b=proximal(groups,nc,XX,XY,tol,L,l2,func);
                yhatbig1=xtest*b+mtrain;
                r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                yhatbig1=xoos*b+mtrain;
                r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=xtrain*b+mtrain;
                r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            end
            
            r2_oos(modeln)=r(2,int16(fw1(r(1,:))));
            r2_is(modeln)=r(3,int16(fw1(r(1,:))));
            l2=10^lamv(int16(fw1(r(1,:))));
            func=@soft_threshodr;
            b=proximal(groups,nc,XX,XY,tol,L,l2,func);
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Ridge R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            func=@soft_threshodr;
            b=proximalH(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,func);
            yhatbig1=xoos*b+mtrain;
            r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=xtrain*b+mtrain;
            r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Ridge+H R2 : ',num2str(r2_oos(modeln),3)));
            
            
            
            %%% Elastic Net %%%
            modeln=modeln+1;
            lamv=-2:0.1:4;
            alpha=0.5;
            r=zeros(3,length(lamv));
            
            for j=1:length(lamv)
                l2=10^lamv(j);
                func=@soft_threshode;
                b=proximal(groups,nc,XX,XY,tol,L,l2,func);
                yhatbig1=xtest*b+mtrain;
                r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                yhatbig1=xoos*b+mtrain;
                r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=xtrain*b+mtrain;
                r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            end
            
            r2_oos(modeln)=r(2,int16(fw1(r(1,:))));
            r2_is(modeln)=r(3,int16(fw1(r(1,:))));
            l2=10^lamv(int16(fw1(r(1,:))));
            func=@soft_threshode;
            b=proximal(groups,nc,XX,XY,tol,L,l2,func);
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Enet R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            func=@soft_threshode;
            b=proximalH(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,func);
            yhatbig1=xoos*b+mtrain;
            r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=xtrain*b+mtrain;
            r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Enet+H R2 : ',num2str(r2_oos(modeln),3)));
            
            
            
            %%% Oracle Models %%%
            modeln=modeln+1;
            if mo==1
                x=zeros(size(xtrain,1),3);
                x(:,1)=xtrain(:,1);
                x(:,2)=xtrain(:,2);
                x(:,3)=xtrain(:,nump+3);
                x1=zeros(size(xoos,1),3);
                x1(:,1)=xoos(:,1);
                x1(:,2)=xoos(:,2);
                x1(:,3)=xoos(:,nump+3);
                
                clf=fitlm(x,ytrain,'Intercept',false);
                yhatbig1=predict(clf,x1);
                r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=predict(clf,x);
                r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
                disp(strcat('Oracle R2 : ',num2str(r2_oos(modeln),3)));
            end
            
            if mo==2
                x=zeros(size(xtrain,1),3);
                x(:,1)=power(xtrain(:,1),2);
                x(:,2)=xtrain(:,2).*xtrain(:,1);
                x(:,3)=sign(xtrain(:,nump+3));
                x1=zeros(size(xoos,1),3);
                x1(:,1)=power(xoos(:,1),2);
                x1(:,2)=xoos(:,2).*xoos(:,1);
                x1(:,3)=sign(xoos(:,nump+3));
                
                clf=fitlm(x,ytrain,'Intercept',false);
                yhatbig1=predict(clf,x1);
                r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=predict(clf,x);
                r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
                disp(strcat('Oracle R2 : ',num2str(r2_oos(modeln),3)));
            end
            
            
            %%% Group Lasso %%%
            kn=4;
            th=zeros(kn,size(xtrain,2));
            th(2,:)=0;
            for i=1:size(xtrain,2)
                th(:,i)=quantile(xtrain(:,i),(0:kn-1)/kn);
            end
            xtrain=cut_knots_degree2(xtrain,kn,th);
            xtest=cut_knots_degree2(xtest,kn,th);
            xoos=cut_knots_degree2(xoos,kn,th);
            
            for i=1:size(xtrain,2)
                s=std(xtrain(:,i));
                if s>0
                    xtrain(:,i)=xtrain(:,i)/s;
                    xtest(:,i)=xtest(:,i)/s;
                    xoos(:,i)=xoos(:,i)/s;
                end
            end
            
            Y=ytrain_demean;
            XX=xtrain.'*xtrain;
            [U,S,V]=svd(XX);
            L=S(1);
            %disp 'L = '
            %disp(L)
            XY=xtrain.'*Y;
            
            modeln=modeln+1;
            lamv=0.5:0.1:3;
            nc=(size(XX,2))/(kn+1);
            groups=repelem(1:nc,kn+1);
            r=zeros(3,length(lamv));
            
            for j=1:length(lamv)
                l2=10^lamv(j);
                func=@soft_threshodg;
                b=proximal(groups,nc,XX,XY,tol,L,l2,func);
                yhatbig1=xtest*b+mtrain;
                r(1,j)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                yhatbig1=xoos*b+mtrain;
                r(2,j)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                yhatbig1=xtrain*b+mtrain;
                r(3,j)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            end
            
            r2_oos(modeln)=r(2,int16(fw1(r(1,:))));
            r2_is(modeln)=r(3,int16(fw1(r(1,:))));
            l2=10^lamv(int16(fw1(r(1,:))));
            
            func=@soft_threshodg;
            b=proximal(groups,nc,XX,XY,tol,L,l2,func);
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Group Lasso R2 : ',num2str(r2_oos(modeln),3)));
            
            
            modeln=modeln+1;
            func=@soft_threshodg;
            b=proximalH(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,func);
            yhatbig1=xoos*b+mtrain;
            r2_oos(modeln)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
            yhatbig1=xtrain*b+mtrain;
            r2_is(modeln)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
            pathb=sprintf('%s',title,'/B/b');
            pathb=sprintf('%s%d_%d_%d',pathb,mo,M,modeln);
            pathb=sprintf('%s',pathb,'.csv');
            csvwrite(pathb,b);
            disp(strcat('Group Lasso+H R2 : ',num2str(r2_oos(modeln),3)));
            
            
            %disp(r2_oos)
            pathr=sprintf('%s',title,'/roos');
            pathr=sprintf('%s_%d_%d',pathr,mo,M);
            pathb=sprintf('%s',pathr,'.csv');
            csvwrite(pathr,r2_oos);
            
            %disp(r2_is)
            pathr=sprintf('%s',title,'/ris');
            pathr=sprintf('%s_%d_%d',pathr,mo,M);
            pathb=sprintf('%s',pathr,'.csv');
            csvwrite(pathr,r2_is);
        end
    end
    
    
end



