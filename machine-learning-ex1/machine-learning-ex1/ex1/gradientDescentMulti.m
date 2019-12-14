function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta0=theta(1,:);
theta1=theta(2,:);
theta2=theta(3,:);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


temp0=theta0-alpha*sum(X*theta-y)/m;
temp1=theta1-alpha*sum((X*theta-y).*X(:,2))/m;
temp2=theta2-alpha*sum((X*theta-y).*X(:,3))/m;
     theta0=temp0;
     theta1=temp1;
     theta2=temp2;








    % ============================================================

    % Save the cost J in every iteration    
    theta=[theta0;theta1;theta2]
    J_history(iter) = 1/(2*m)*sum((X*theta-y).^2);

end

end
