function [J, grad] = costFunctionReg(theta, X, Y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(Y); % number of training examples

% You need to return the following variables correctly
z=X*theta;
h=1./(1+exp(-z));
theta1=theta((2:400),:)
J=(-Y'*log(h)-(1-Y)'*log(1-h))*(1/m)+(lambda/(2*m))*sum(theta1.^2);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

  
  temp1=(1/m)*sum((h-Y).*X(:,1)
  
  
  
 temp2=(1/m)*sum((X(2:5000)'*(h-Y)+(lambda/m)*theta(2:400);
  grad=[temp1 temp2];





% =============================================================

end
