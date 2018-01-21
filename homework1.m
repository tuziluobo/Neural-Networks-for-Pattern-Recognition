trainset=loadMNISTImages('train-images.idx3-ubyte');
trainset=trainset(:,1:20000);
trainset0=zeros(785,20000);
trainset0(1,1:20000)=1;
trainset0(2:785,1:20000)=trainset;
trainset=trainset0;
hold_out=trainset(:,1:2000);
trainset=trainset(:,2001:20000);
trainlabel=loadMNISTLabels('train-labels.idx1-ubyte');
trainlabel=trainlabel(1:20000,1);
holdlabel=trainlabel(1:2000,1);
trainlabel=trainlabel(2001:20000,1);
testset=loadMNISTImages('t10k-images.idx3-ubyte');
testset=testset(:,8001:10000);
testset0=zeros(785,2000);
testset0(1,:)=1;
testset0(2:785,:)=testset;
testset=testset0;
testlabel=loadMNISTLabels('t10k-labels.idx1-ubyte');
testlabel=testlabel(8001:10000,1);
X=[];
target=[];
X_hold=[];
target_hold=[];
X_t=[];
target_t=[];
t=1;
for i=1:size(holdlabel)
    if(holdlabel(i)==2)
        X_hold(:,t)=hold_out(:,i);
        target_hold(t)=1;
        t=t+1;
    elseif(holdlabel(i)==3)
        X_hold(:,t)=hold_out(:,i);
        target_hold(t)=0;
        t=t+1;
    end
end
t=1;
for i=1:size(trainlabel)
    if(trainlabel(i)==2)
        X(:,t)=trainset(:,i);
        target(t)=1;
        t=t+1;
    elseif(trainlabel(i)==3)
        X(:,t)=trainset(:,i);
        target(t)=0;
        t=t+1;
    end
end
t=1;
for i=1:size(testlabel)
    if(testlabel(i)==2)
        X_t(:,t)=testset(:,i);
        target_t(t)=1;
        t=t+1;
    elseif(testlabel(i)==3)
        X_t(:,t)=testset(:,i);
        target_t(t)=0;
        t=t+1;
    end
end
w=zeros(1,785);
[w,accuracy]=LogisticR(0.00001,w,X,target,X_hold,target_hold,X_t,target_t);
w28=w;
%w28=w(2:785);
%w28=reshape(w28,[28,28]);




