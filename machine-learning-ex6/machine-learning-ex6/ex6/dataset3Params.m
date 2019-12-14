function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
x1 = [1 2 1]; x2 = [0 4 -1];
v=[0.01 0.03 0.1 0.3 1 3 10 30];
a=1;
temp=zeros(length(v)*length(v),3)
for i=1:length(v)
  ct=v(i);
  for j=1:length(v)
    sigmat=v(j);
model= svmTrain(X, y, ct, @(x1, x2) gaussianKernel(x1, x2, sigmat));
predictions = svmPredict(model, Xval);
error=mean(double(predictions ~= yval));
temp(a,1)=v(i);
temp(a,2)=v(j);
temp(a,3)=error;
a=a+1;
end
end
[val,ind]=min(temp(:,3));
C=temp(ind,1)
sigma=temp(ind,2)

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
