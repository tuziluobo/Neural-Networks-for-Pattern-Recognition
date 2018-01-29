trainset=loadMNISTImages('train-images.idx3-ubyte');
trainlabel=loadMNISTLabels('train-labels.idx1-ubyte');
testset=loadMNISTImages('t10k-images.idx3-ubyte');
testlabel=loadMNISTLabels('t10k-labels.idx1-ubyte');
trainset=trainset*255/127.5-1;
testset=testset*255/127.5-1;
r=randperm(size(trainset, 2)); %shuffle
trainset=trainset(:,r);
trainlabel=trainlabel';
trainlabel=trainlabel(:,r);
holdset=trainset(:,50001:60000);
holdlabel=trainlabel(:,50001:60000);
trainset=trainset(:,1:50000);
trainlabel=trainlabel(:,1:50000);
w1=normrnd(0,1/28/28,28*28,64); %initialize
w2=normrnd(0,1/64,64,10);
[error_te]=neuralnetwork(trainset, trainlabel, holdset, holdlabel, testset, testlabel, w1,w2);






