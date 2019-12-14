function [J, grad] = costFunction(theta, X, Y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values % number of training examples

% You need to return the following variables correctly 

m = length(Y); % number of training examples
z=X*theta;
h=1./(1+exp(-z))
size(h)
J=(-Y'*log(h)-(1-Y)'*log(1-h))*(1/m)

% You need to return the following variables correctly 



  temp1=(1/m)*sum((h-Y).*X(:,1));
  temp2=(1/m)*sum((h-Y).*X(:,2));
  temp3=(1/m)*sum((h-Y).*X(:,3));
  theta1=temp1;
  theta2=temp2;
  theta3=temp3;
  grad=[theta1;theta2;theta3];
 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
