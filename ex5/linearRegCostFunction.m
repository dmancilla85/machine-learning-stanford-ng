function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% H0 function
h0 = X * theta;

% Theta dimension
lt = length(theta);

% Cost function
J = 1/(2*m)*(sum((h0 - y).^2) + lambda * sum(theta(2:lt).^2));

% First term of gradient without regularization
grad = (1/m) * X' * (h0 - y);

% Add regularization term just for theta(1:m)
grad(2:lt) += (lambda/m)*theta(2:lt);

% =========================================================================
grad = grad(:);

end
