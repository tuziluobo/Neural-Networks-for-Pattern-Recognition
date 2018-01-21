function [w,accuracy] = LogisticRL2(sigma,w,X,target,X_hold,target_hold,X_t,target_t) 
%calculate w using L2, input sigma is learning rate
lamda=[0.001,0.0001,0.00001,0.000001];
k=0;
hold_error0=0;
t=1;
error_train=[];
accuracy_train=[];
error_hold=[];
accuracy_hold=[];
error_test=[];
accuracy_test=[];
rate=sigma;
T=10000;
error_test=[];
j=1;
%for j=1:4
    accuracy_train=[];
    length=[];
    w=zeros(1,785);
    t=1;
    w0=zeros(1,785);
    k=0;
    while(t<5000)
        y=1./(1+exp(-w*X));
        [accuracy,gradient,error]=LogisticError(y,X,target);    
        %LogisticError() to calculate gradient and error
        accuracy_train(t)=accuracy;
        error_train(t)=error;
        w=w+rate.*(gradient-lamda(j)*size(X,2)*2*w);
        %update w
        %w=w+rate.*(gradient-size(X,2)*lamda(j)*w0);
        %update w
        %w=w+rate.*gradient;            
        %update w
        length(t)=norm(w,2);
        %calculate length of w
        w0(find(w>0))=1;
        w0(find(w<=0))=-1;
        y_hold=1./(1+exp(-w*X_hold));
        y_test=1./(1+exp(-w*X_t));
        [a_h,hold_gradient,hold_error]=LogisticError(y_hold,X_hold,target_hold);   
        [a_t,test_gradient,test_error]=LogisticError(y_test,X_t,target_t);
        accuracy_hold(t)=a_h;
        error_hold(t)=hold_error;
        accuracy_test(t)=a_t;
        error_test(t)=test_error;
        if(hold_error>=hold_error0)
            k=k+1;
        else
           k=0;
        end
      if(k>=20)   
          %early stopping
            break;
      end
        t=t+1;
        hold_error0=hold_error;
        rate=sigma/(1+(t-1)/T);
    end
%end
%y_test=1./(1+exp(-w*X_t));
%[accu_t,gradient_t,error_t]=LogisticError(y_test,X_t,target_t);
%error_test(t)=error_t;
%x0=1:t;
%plot(x0,accuracy_train);
%hold on;
%hold on;
%plot(x0,accuracy_hold);
%hold on;
%plot(x0,accuracy_test);
%end
%figure;
%plot(x0,length);
%hold on;
%hold on;
%figure;
%end
%figure;
%plot(log10(lamda),error_test);
w23=w(1,2:785);
wmax=max(w23);
wmin=min(w23);
w23=(w23-wmin)/(wmax-wmin);
w23=reshape(w23,[28,28]);
imagesc(abs(w23));


