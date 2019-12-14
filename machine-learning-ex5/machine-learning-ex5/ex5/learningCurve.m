function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
%m = size(X, 1);
%mcv=size(Xval,1);
m=size(X,1);
m1=size(Xval,1);
% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
for i=1:m
  [theta] = trainLinearReg( X(1:i,:), y(1:i,:), lambda);
  h=X(1:i,:)*theta;
  h1=Xval*theta;

  error_train(i)=(1/(2*i))*sum((h-y(1:i)).^2)%+(lambda/(2*m))*sum(theta([2:end],:).^2);
error_val(i)=(1/(2*m1))*sum((h1-yval).^2)%+(lambda/(2*m))*sum(theta([2:end],:).^2);
endfor
error_train
error_val
%for i=1:m
 % h=X(1:i,:)*theta;
 % h1=Xval(1:i,:)*theta;
 % error_train(i)=(1/(2*m))*sum((h-y(1:i)).^2);%+(lambda/(2*m))*sum(theta([2:end],:).^2);
  %[theta] = trainLinearReg(X, y, lambda);
%end
%[theta] = trainLinearReg(X, y, lambda);
%for j=1:size(Xval)
  %h1=Xval(1:j,:)*theta;
 % error_val(j)  =(1/(2*mcv))*sum((h1-yval(1:j)).^2);%+(lambda/(2*m))*sum(theta([2:end],:).^2);
 % end
 %error_train
 %error_val
 
% plot(1:m, error_train, 1:size(Xval), error_val);
%title('Learning curve for linear regression')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Error')
%axis([0 13 0 150])
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------







% -------------------------------------------------------------

% =========================================================================

end