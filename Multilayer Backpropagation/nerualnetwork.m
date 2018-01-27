function [y2] = neuralnetwork(trainset, trainlabel, holdset, holdlabel, w1,w2)
rate=0.0001;
num=0;
epoch=0;
trainerror=[];
holderror=[];
trainaccu=[];
while(1)
    for k=1:size(trainset,2)/128
        minibatch=trainset(:,(k-1)*128+1:k*128);
        minilabel=trainlabel(:,(k-1)*128+1:k*128);
        a1=w1'*minibatch;
        z=1./(1+exp(-a1));
        a2=w2'*z;
        sumexp=sum(exp(a2));
        y=exp(a2)./sumexp;
        t=zeros(10,128);
        for i=1:128
            t(minilabel(i)+1,i)=1;
        end
        delta1=z.*(1-z).*sum(w2*(t-y));
        w1=w1+rate*minibatch*delta1';
        w2=w2+rate*z*(t-y)';
    end
    a1_h=w1'*holdset;
    z_h=1./(1+exp(-a1_h));
    a2_h=w2'*z_h;
    sumexp_h=sum(exp(a2_h));
    y_h=exp(a2_h)./sumexp_h;
    t_h=zeros(10,10000);
    for i=1:10000
        t_h(holdlabel(i)+1,i)=1;
    end
    error_h=-sum(sum(log(y_h).*t_h));
    if(epoch==0)
       error_before=error_h;
    elseif(error_before<error_h&&num>5)
        break;
    elseif(error_before<error_h)
        num=num+1;
    else
        num=0;
    end
    error_before=error_h;
    a1_t=w1'*trainset;
    z_t=1./(1+exp(-a1_t));
    a2_t=w2'*z_t;
    sumexp_t=sum(exp(a2_t));
    y_t=exp(a2_t)./sumexp_t;
    t_t=zeros(10,50000);
    accu=0;
    for i=1:50000
        res=max(y_t(:,i));
        index=find(y_t(:,i)==res)-1;
        if(trainlabel(i)==index)
            accu=accu+1;
        end
        t_t(trainlabel(i)+1,i)=1;
    end
    accu_h=0;
    for i=1:10000
        res_h=max(y_h(:,i));
        index_h=find(y_h(:,i)==res_h)-1;
        if(holdlabel(i)==index_h)
            accu_h=accu_h+1;
        end
        t_h(holdlabel(i)+1,i)=1;
    end
     epoch=epoch+1;
     trainaccu(epoch)=accu/50000;
     r=randperm(size(trainset, 2));
     trainset=trainset(:,r);
     trainlabel=trainlabel(:,r);
     holdaccu(epoch)=accu_h/10000;
end
plot(1:epoch,trainaccu);
figure;
plot(1:epoch,holdaccu);


