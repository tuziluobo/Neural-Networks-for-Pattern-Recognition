function [error_te] = momentum(trainset, trainlabel, holdset, holdlabel, testset, testlabel, w1,w2)
v1=zeros(28*28,64);
v2=zeros(64,10);
alpha=0.9;  %alpha of momentum
rate=0.0001;
num=0;
epoch=0;
trainerror=[];
holderror=[];
trainaccu=[];
for i=1:10000
    t_h(holdlabel(i)+1,i)=1;
    t_te(testlabel(i)+1,i)=1;
end
while(1)
    for k=1:size(trainset,2)/128
        minibatch=trainset(:,(k-1)*128+1:k*128);
        minilabel=trainlabel(:,(k-1)*128+1:k*128);
        a1=w1'*minibatch;
        z=1.7159.*tanh(2/3.*a1);
        a2=w2'*z;
        sumexp=sum(exp(a2));
        y=exp(a2)./sumexp;
        t=zeros(10,128);
        for i=1:128
            t(minilabel(i)+1,i)=1;
        end
        unit=ones(64,1);
        delta1=1.7159*2/3*(unit-z.*z./1.7159./1.7159).*(w2*(t-y));
        v1=alpha*v1+rate*minibatch*delta1';   %momentum
        w1=w1+v1;
        v2=alpha*v2+rate*z*(t-y)';
        w2=w2+v2;
    end
    accu_h=0;
    a1_h=w1'*holdset;
    z_h=1.7159.*tanh(2/3.*a1_h);
    a2_h=w2'*z_h;
    sumexp_h=sum(exp(a2_h));
    y_h=exp(a2_h)./sumexp_h;
    
    accu_te=0;       %accuracy of test set
    a1_te=w1'*testset;
    z_te=1.7159.*tanh(2/3.*a1_te);
    a2_te=w2'*z_te;
    sumexp_te=sum(exp(a2_te));
    y_te=exp(a2_te)./sumexp_te;    
    for i=1:10000
        res_h=max(y_h(:,i));
        res_te=max(y_te(:,i));
        index_h=find(y_h(:,i)==res_h)-1;
        index_te=find(y_te(:,i)==res_te)-1;
        if(holdlabel(i)==index_h)
            accu_h=accu_h+1;
        end
        if(testlabel(i)==index_te)
            accu_te=accu_te+1;
        end
    end
    error_h=-sum(sum(log(y_h).*t_h));
    error_te=-sum(sum(log(y_te).*t_te));
    if(epoch==0)
       error_before=error_h;
    elseif(error_before<error_h&&num>=2)
        break;
    elseif(error_before<error_h)
        num=num+1;
    else
        num=0;
    end   
    error_before=error_h;
    
    a1_t=w1'*trainset;
    z_t=1.7159.*tanh(2/3.*a1_t);
    a2_t=w2'*z_t;
    sumexp_t=sum(exp(a2_t));
    y_t=exp(a2_t)./sumexp_t;
    t_t=zeros(10,50000);
    accu_t=0;    %accuracy of train set
    
    for i=1:50000
         res=max(y_t(:,i));
         index=find(y_t(:,i)==res)-1;
         if(trainlabel(i)==index)
             accu_t=accu_t+1;
         end
         t_t(trainlabel(i)+1,i)=1;
    end
    error_t=-sum(sum(log(y_t).*t_t));
    accu_h=0;
    accu_te=0;
    for i=1:10000
        res_h=max(y_h(:,i));
        res_te=max(y_te(:,i));
        index_h=find(y_h(:,i)==res_h)-1;
        index_te=find(y_te(:,i)==res_te)-1;
        if(holdlabel(i)==index_h)
            accu_h=accu_h+1;
        end
        if(testlabel(i)==index_te)
            accu_te=accu_te+1;
        end
    end
    epoch=epoch+1;
    holderror(epoch)=error_h;
    testerror(epoch)=error_te;
    trainerror(epoch)=error_t;
    trainaccu(epoch)=accu_t/50000;
    holdaccu(epoch)=accu_h/10000;
    testaccu(epoch)=accu_te/10000;
    r=randperm(size(trainset, 2));
    trainset=trainset(:,r);
    trainlabel=trainlabel(:,r);
end
plot(1:epoch,trainaccu);
hold on;
plot(1:epoch,holdaccu);
hold on;
plot(1:epoch,testaccu);
figure;
plot(1:epoch,trainerror);
hold on;
plot(1:epoch,holderror);
hold on;
plot(1:epoch, testerror);

