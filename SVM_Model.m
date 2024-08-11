% Load the dataset
data = readtable('motor_fault_dataset_with_fault_codes2.xlsx');

% Separate features (X) and labels (y)
X = data(:, 1:end-1);
y = data.Fault_Type;

% Split the dataset into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Manually Standardize the data
mu = mean(table2array(X_train));
sigma = std(table2array(X_train));
X_train_scaled = (table2array(X_train) - mu) ./ sigma;
X_test_scaled = (table2array(X_test) - mu) ./ sigma;

% Define the best parameters
C = 10;
gamma = 0.1;
kernelFunction = 'rbf';

% Train the SVM model using fitcecoc for multi-class classification
template = templateSVM('KernelFunction', kernelFunction, ...
                       'KernelScale', 1/gamma, ...
                       'BoxConstraint', C);

SVMModel = fitcecoc(X_train_scaled, y_train, 'Learners', template);

% Make predictions on the test set
y_pred = predict(SVMModel, X_test_scaled);

% Evaluate the model with confusion matrix and classification report
confMat = confusionmat(y_test, y_pred);
disp('Confusion Matrix:');
disp(confMat);

% Calculate additional metrics as needed
disp('Classification Report:');
classMetrics = classperf(y_test, y_pred);
disp(classMetrics);

% Predicting the fault type for new data
new_data = table(280, 280, 280, 50, 50, 50, 200, 1000, 0, 50, ...
    'VariableNames', {'Voltage_Phase_A', 'Voltage_Phase_B', ...
                      'Voltage_Phase_C', 'Current_Phase_A', ...
                      'Current_Phase_B', 'Current_Phase_C', ...
                      'Torque', 'RPM', 'Vibration_Amplitude', ...
                      'Vibration_Frequency'});

% Standardize the new data manually
new_data_scaled = (table2array(new_data) - mu) ./ sigma;

% Predict the fault type
predicted_fault = predict(SVMModel, new_data_scaled);
disp(['Predicted Fault Type for new data: ', char(predicted_fault)]);

% Save the trained model and scaler parameters
save('svm_model_best_params.mat', 'SVMModel');
save('scaler_params.mat', 'mu', 'sigma');
disp('Model and Scaler Parameters Exported successfully');


% Plot the confusion matrix as a heatmap
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix');

% Calculate the scores for each class
[~, scores] = predict(SVMModel, X_test_scaled);

% Calculate the ROC curve for each class
figure;
for i = 1:numel(unique(y_test))
    [Xroc, Yroc, T, AUC] = perfcurve(y_test, scores(:, i), SVMModel.ClassNames{i});
    plot(Xroc, Yroc);
    hold on;
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
legend(SVMModel.ClassNames, 'Location', 'Best');
hold off;

% Calculate the precision-recall curve for each class
figure;
for i = 1:numel(unique(y_test))
    [Xpr, Ypr, ~, AUC] = perfcurve(y_test, scores(:, i), SVMModel.ClassNames{i}, 'xCrit', 'reca', 'yCrit', 'prec');
    plot(Xpr, Ypr);
    hold on;
end
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curve');
legend(SVMModel.ClassNames, 'Location', 'Best');
hold off;

% Plot actual vs. predicted labels as categorical values
figure;
subplot(2,1,1);
plot(categorical(y_test), 'o', 'DisplayName', 'Actual', 'MarkerSize', 8);
hold on;
plot(categorical(y_pred), 'x', 'DisplayName', 'Predicted', 'MarkerSize', 8);
legend('Location', 'Best');
xlabel('Sample Index');
ylabel('Fault Type');
title('Actual vs. Predicted Fault Types');
hold off;

% Plot mismatch between actual and predicted
subplot(2,1,2);
mismatch = ~strcmp(y_test, y_pred);
plot(mismatch, 'r.', 'MarkerSize', 10);
ylim([-0.5 1.5]);
xlabel('Sample Index');
ylabel('Mismatch (1 = Error)');
title('Prediction Errors');


% Calculate the scores for each class
[~, scores] = predict(SVMModel, X_test_scaled);

% Plot model response (scores) for each class
figure;
for i = 1:numel(SVMModel.ClassNames)
    subplot(numel(SVMModel.ClassNames), 1, i);
    plot(scores(:, i), 'o-', 'DisplayName', ['Score for ', char(SVMModel.ClassNames{i})]);
    hold on;
    plot(double(strcmp(y_test, SVMModel.ClassNames{i})), 'x-', 'DisplayName', 'True Label');
    legend('Location', 'Best');
    xlabel('Sample Index');
    ylabel('Score');
    title(['Model Response: ', char(SVMModel.ClassNames{i})]);
    hold off;
end

% Convert true and predicted labels to numerical indices
[~, y_test_num] = ismember(y_test, SVMModel.ClassNames);
[~, y_pred_num] = ismember(y_pred, SVMModel.ClassNames);

% Calculate residuals (difference between true and predicted values)
residuals = y_test_num - y_pred_num;

% Plot residuals
figure;
stem(residuals, 'MarkerFaceColor', 'r');
xlabel('Sample Index');
ylabel('Residuals (True - Predicted)');
title('Residual Plot for Classification');

% Assuming you have only two features for simplicity
figure;
gscatter(X_test_scaled(:,1), X_test_scaled(:,2), y_test);
hold on;

% Generate a grid of values to evaluate the decision boundary
[x1Grid, x2Grid] = meshgrid(linspace(min(X_test_scaled(:,1)), max(X_test_scaled(:,1)), 100), ...
                            linspace(min(X_test_scaled(:,2)), max(X_test_scaled(:,2)), 100));
xGrid = [x1Grid(:), x2Grid(:)];

% Predict over the grid
[~, scores] = predict(SVMModel, xGrid);

% Plot decision boundary
contour(x1Grid, x2Grid, reshape(scores(:,classIdx), size(x1Grid)), [0, 0], 'k');
xlabel('Feature 1');
ylabel('Feature 2');
title('Decision Boundary for "Locked Rotor"');
hold off;
