function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

log_h = log(sigmoid(X*theta));
log_1_h = log(1 - sigmoid(X*theta));

% Theta dimension
lt = length(theta);

J = (1/m) * (-y' * log_h - (1-y)' * log_1_h) + (lambda/(2*m)) * sum(theta(2:lt).^2);

grad = (1/m) * X' * (sigmoid(X*theta) - y);

% Add regularization term just for theta(1:m)
grad(2:lt) += (lambda/m)*theta(2:lt);

% =============================================================

end
