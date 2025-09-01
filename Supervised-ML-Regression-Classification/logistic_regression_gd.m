
clc; clear; close all;

% --- Load data ---
data = load('data2.txt');
X = data(:, 1:2);
y = data(:, 3);

% --- Hyperparameters / init ---
w = [0.01; 0.01];                     % initial weights (2x1)
b = -8;                               % initial bias
alpha = 1e-3;                         % learning rate
num_iters = 10000;                    % iterations
print_every = 200;                    % console logging cadence

% --- Train (Gradient Descent) ---
[w, b, J_history] = gradient_descent(X, y, w, b, alpha, num_iters, print_every);

% --- Train accuracy ---
p = predict(X, w, b);
fprintf('Train Accuracy: %.2f%%\n', mean(double(p == y)) * 100);

% --- Plots ---
% Data + decision boundary
figure; hold on; grid on;
gscatter(X(:,1), X(:,2), y, 'kr', '+o');
xlabel('Exam 1 score'); ylabel('Exam 2 score');
title('Data by class and decision boundary');
legend('Admitted','Not admitted','Location','northeast');

% Decision boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(b + w1*x1)/w2
x1_vals = linspace(min(X(:,1))-1, max(X(:,1))+1, 200);
x2_vals = -(b + w(1)*x1_vals)/w(2);
plot(x1_vals, x2_vals, 'b-', 'LineWidth', 2);
hold off;

% =============== Helper functions ===============

function p = predict(X, w, b)
    % Vectorized prediction: p = 1 if sigmoid(Xw+b) >= 0.5
    z = X*w + b;                       
    f = 1 ./ (1 + exp(-z));            
    p = double(f >= 0.5);
end

function [w, b, J_history] = gradient_descent(X, y, w, b, alpha, num_iters, print_every)
    J_history = zeros(num_iters, 1);
    for i = 1:num_iters
        [dj_db, dj_dw] = compute_gradient(X, y, w, b);
        b = b - alpha * dj_db;
        w = w - alpha * dj_dw;

        J_history(i) = compute_cost(X, y, w, b);

        if mod(i, print_every) == 0 || i == 1
            fprintf('Iter %5d | Cost = %.6f\n', i, J_history(i));
        end
    end
end

function [dj_db, dj_dw] = compute_gradient(X, y, w, b)
    m = size(X, 1);
    z = X*w + b;                       
    f = 1 ./ (1 + exp(-z));            
    err = f - y;                       

    dj_db = sum(err) / m;              
    dj_dw = (X' * err) / m;            
end

function J = compute_cost(X, y, w, b)
    m = size(X, 1);
    z = X*w + b;
    f = 1 ./ (1 + exp(-z));            

    eps_ = 1e-15;                      
    f = min(max(f, eps_), 1 - eps_);

    J = -(1/m) * sum( y .* log(f) + (1 - y) .* log(1 - f) );
end
