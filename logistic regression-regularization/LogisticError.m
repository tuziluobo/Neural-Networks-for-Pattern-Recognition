function [accuracy,gradient,error] = LogisticError(y,X,target)
%calculate gradient, error, and percent of correctness
accuracy=0;
gradient=-(y-target)*X';
[m,n]=size(y);
error=0;
for i=1:size(target,2)
    error=error+target(i)*log(y(i))+(1-target(i))*log(1-y(i));
    if((y(i)<0.5&&target(i)==0)||(y(i)>=0.5&&target(i)==1))
        accuracy=accuracy+1;
    end
end
accuracy=accuracy/n;
error=-error/n;

