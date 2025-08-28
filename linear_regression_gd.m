
clc; clear; close all;

% --- Load data ---
data = load('data1.txt');
x = data(:,1);
y = data(:,2);

% --- Hyperparameters ---
initial_w   = 0.0;
initial_b   = 0.0;
iterations  = 1500;
alpha       = 0.01;

% --- Train model ---
[w, b] = gradient_descent(x, y, initial_w, initial_b, alpha, iterations);

% --- Predictions for training data ---
predicted = w * x + b;

% --- Plot results ---
figure;
plot(x, y, 'xr', 'MarkerSize', 8, 'LineWidth', 1.5); hold on;
plot(x, predicted, 'b', 'LineWidth', 1.5);
title('Profits vs. Population per City');
xlabel('Population of City (×10,000s)');
ylabel('Profit (×$10,000)');
legend({'Training Data', 'Linear Regression'}, 'Location', 'best');
grid on;

% --- Example predictions ---
predict1 = 3.5 * w + b;
fprintf('For population = 35,000, predicted profit = $%.2f\n', predict1 * 1e4);

predict2 = 7.0 * w + b;
fprintf('For population = 70,000, predicted profit = $%.2f\n', predict2 * 1e4);


function [w,b] = gradient_descent(x, y, w_in, b_in, alpha, num_iters)

 m = length(x);

 J_history = zeros(num_iters,1);
 w_history = zeros(num_iters,1);

 w = w_in;
 b = b_in;

 prints = 0:150:num_iters;
    
 for i = 1:num_iters
     [dj_db, dj_dw] = compute_gradient(x, y, w, b);
    
     w = w - alpha * dj_dw;               
     b = b - alpha * dj_db;

     cost = compute_cost(x, y, w, b);
     J_history(i) = cost;

     if ismember(i, prints)
         w_history(i) = w;
         fprintf('Iter %4d | Cost = %.6f | w = %.4f | b = %.4f\n', i, cost, w, b);
     end
 end

 fprintf('Training complete. Final parameters: w = %.6f | b = %.6f\n', w, b);
end


function j = compute_cost(x, y, w, b)
    m = length(x);
    f = w * x + b;
    j = sum((f - y).^2)/(2*m);
end


function [dj_db, dj_dw] = compute_gradient(x, y, w, b)
    m = length(x);
    f = w * x + b;
    dj_db = sum((f - y))/m;
    dj_dw = sum((f - y) .* x)/m;
end
