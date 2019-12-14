function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

 % Number of training examples

% You need to return the following variables correctly


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
prob = sigmoid(X * theta);
if ( prob >= 0.5)
  p=1
else
  p = 0
 endif
 data=load('ex2data1.txt');
 m = length(data(:,1));
 Y=data(:,3);
 %z= [ones(m,1) data(:,[1,2])];
 prob1 = sigmoid(z * theta)
 z=zeros(m,1);
 for i=1:m
   if(prob1(i)>=0.5)
    z(i)=1;
  else
    z(i)=0;
  endif
  end
   
accuracY=mean(double(z == Y)) * 100







% =========================================================================


end
