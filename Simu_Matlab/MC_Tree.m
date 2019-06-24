%%% This Version: July 30, 2018. @copyright Shihao Gu, Bryan Kelly and Dacheng Xiu
%%% If you use these codes, please cite the paper "Empirical Asset Pricing via Machine Learning" (2018) and "Autoencoder Asset Pricing Models." (2019)

%%% All tree models %%%

MC=1; % setup MC number
datanum=50; %Or datanum=100; seperately run two cases
path='./Simu'; % set your own folder path    
dirstock=strcat(path,'/SimuData_p',int2str(datanum),'/');

for hh=[1]
%for hh = [1 3 6 12] % correspond to monthly quarterly halfyear and annually returns

    title=strcat(path,'/Simu_p',int2str(datanum),'/Tree',int2str(hh));

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
            ind=floor(N*T/3)+1:floor(N*(T*2/3+1));
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
            
            
            %%% Start to train %%%
            
            r2_oos=zeros(3,1);  %%% OOS R2
            r2_is=zeros(3,1);  %%% IS R2
            
            
            %%% Random Forest %%%
            if nump == 50
                lamv = 10:10:100;
            end
            if nump == 100
                lamv = 10:20:200;
            end
            ne=100;
            lamc = [2,4,8,16,32];
            r=zeros(length(lamv),length(lamc),3);
            
            for n1 = 1:length(lamv)
                nf=lamv(n1);
                for n2 = 1:length(lamc)
                    nn=lamc(n2);
                    clf=TreeBagger(ne,xtrain,ytrain,'Method','regression','NumPredictorsToSample',nf,'MaxNumSplits',nn);
                    yhatbig1 = predict(clf,xtest);
                    r(n1,n2,1)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                    yhatbig1 = predict(clf,xoos);
                    r(n1,n2,2)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                    yhatbig1 = predict(clf,xtrain);
                    r(n1,n2,3)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
                end  
            end
            
            fw_2 = fw2(r(:,:,1));
            r2_oos(1)=r(fw_2(1),fw_2(2),2);
            r2_is(1)=r(fw_2(1),fw_2(2),3);
            disp(strcat('RF R2 : ',num2str(r2_oos(1),3)));
            
            
            %%% GBRT %%%
            
            lamv=-1:0.2:0;
            r=zeros(length(lamv),50,3);
            
            for n1 = 1: length(lamv)
                lr=10^lamv(n1);
                alpha=2;
                ne=50;
                t=templateTree('MaxNumSplits',2,'Surrogate','on');
                clf=fitensemble(xtrain,ytrain,'LSBoost',ne,t,'Type','regression','LearnRate',lr);
                
                % e=predict(clf,xtest);
                % e = error(clf,xtest,ytest);
                e=loss(clf,xtest,ytest,'mode','cumulative');
                for i = 1:length(e);
                    r(n1,i,1) = e(i);
                    % pred = e(i);
                    % yhatbig1 = pred;
                    % r(n1,i,1)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
                end
                
                %e=error(clf,xoos,yoos);
                e=loss(clf,xoos,yoos,'mode','cumulative');
                for i = 1:length(e);
                    r(n1,i,2) = e(i);
                    % pred = e(i);
                    % yhatbig1 = pred;
                    % r(n1,i,2)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
                end
                
                %e=error(clf,xtrain,ytrain);
                e=loss(clf,xtrain,ytrain,'mode','cumulative');
                for i = 1:length(e);
                    r(n1,i,3) = e(i);
                    % pred = e(i);
                    % yhatbig1 = pred;
                    % r(n1,i,2)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
                end
                    
            end
            
            fw_2 = fw2(-r(:,:,1));
            err1=mean((ytrain-mtrain).^2);
            err2=mean((yoos-mtrain).^2);
            r2_oos(2)=1-r(fw_2(1),fw_2(2),2)/err2;
            r2_is(2)=1-r(fw_2(1),fw_2(2),3)/err1;
            disp(strcat('GBRT R2 : ',num2str(r2_oos(2),3)));
            
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