function [J, grad] = linearRegCostFunction(X1, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h=X1*theta;
J=(1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*sum(theta([2:end],:).^2)
theta1=(1/m)*sum((h-y).*X1(:,1));
for i=2:size(theta)
theta2(i-1)=(1/m)*sum((h-y).*X1(:,i))+(lambda/m)*theta(i);
end
%theta3=(1/m)*sum((h-y).*X1(:,3))+(lambda/m)*theta(3);
grad=[theta1;theta2']


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
