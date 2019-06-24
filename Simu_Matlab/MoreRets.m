%%% This Version: July 30, 2018. @copyright Shihao Gu, Bryan Kelly and Dacheng Xiu
%%% If you use these codes, please cite the paper "Empirical Asset Pricing via Machine Learning" (2018) and "Autoencoder Asset Pricing Models." (2019)

%%% Generate quarterly halfyear and annually returns %%%

path='./Simu'; % set your own folder path    
name1='/SimuData_p50'; % Case Pc=50 
name2='/SimuData_p100'; % Case Pc=100

for name=string({name1,name2})
    for mo=[1,2]
        for M=1:1
            disp(M)
            
            dirstock=sprintf('%s',path,name);
            dirstock=sprintf('%s',dirstock,'/');
            path2=sprintf('%s',dirstock,'r');
            path2=sprintf('%s%d_%d',path2,mo,M);
            path2=sprintf('%s',path2,'.csv');
            r=csvread(path2,0,0);
            r3=zeros(length(r),1);
            r6=zeros(length(r),1);
            r12=zeros(length(r),1);
            
            per=repmat(1:200,1,180);
            time=repelem(1:180,200);
            u=unique(per);
            for i=1:length(u)
                ind=(per==u(i));
                ret=r(ind);
                
                
                ret3=zeros(length(ret),1);
                N=length(ret3);
                for j=1:(N-2)
                    ret3(j)=sum(ret(j:(j+2)));
                end
                r3(ind)=ret3;
                
                
                ret6=zeros(length(ret),1);
                N=length(ret6);
                for j=1:(N-5)
                    ret6(j)=sum(ret(j:(j+5)));
                end
                r6(ind)=ret6;
                
                
                ret12=zeros(length(ret),1);
                N=length(ret12);
                for j=1:(N-11)
                    ret12(j)=sum(ret(j:(j+11)));
                end
                r12(ind)=ret12;
                
            end
            K=200*180;
            a=(1:K)';
            df=[a;r];
            % df=mat2dataset(df,'VarNames',{'a','r'});
            pathr=sprintf('%s',dirstock,'r');
            pathr=sprintf('%s%d_%d_%d',pathr,mo,M,1);
            pathr=sprintf('%s',pathr,'.csv');
            csvwrite(pathr,df);
            
            df=[a;r3];
            % df=mat2dataset(df,'VarNames',{'a','r3'});
            pathr=sprintf('%s',dirstock,'r');
            pathr=sprintf('%s%d_%d_%d',pathr,mo,M,3);
            pathr=sprintf('%s',pathr,'.csv');
            csvwrite(pathr,df);
            
            df=[a;r6];
            % df=mat2dataset(df,'VarNames',{'a','r6'});
            pathr=sprintf('%s',dirstock,'r');
            pathr=sprintf('%s%d_%d_%d',pathr,mo,M,6);
            pathr=sprintf('%s',pathr,'.csv');
            csvwrite(pathr,df);
            
            df=[a;r12];
            % df=mat2dataset(df,'VarNames',{'a','r12'});
            pathr=sprintf('%s',dirstock,'r');
            pathr=sprintf('%s%d_%d_%d',pathr,mo,M,12);
            pathr=sprintf('%s',pathr,'.csv');
            csvwrite(pathr,df);
        end
    end
end