trainset=loadMNISTImages('train-images.idx3-ubyte');
trainlabel=loadMNISTLabels('train-labels.idx1-ubyte');
trainset=trainset*255/127.5-1;
holdset=trainset(:,50001:60000);
trainlabel=trainlabel';
holdlabel=trainlabel(:,50001:60000);
trainset=trainset(:,1:50000);
trainlabel=trainlabel(:,1:50000);
w1=normrnd(0,1,28*28,64);
w2=normrnd(0,1,64,10);
[y]=neuralnetwork(trainset, trainlabel, holdset, holdlabel,w1,w2);



