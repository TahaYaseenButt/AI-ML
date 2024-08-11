% Load the saved SVM model
load('svm_model_best_params.mat', 'SVMModel');

% Load the saved scaler parameters
load('scaler_params.mat', 'mu', 'sigma');

% Define new data for prediction
new_data = table(220, 230, 224, 50, 50, 50, 190, 1500, 0.005, 70, ...
    'VariableNames', {'Voltage_Phase_A', 'Voltage_Phase_B', ...
                      'Voltage_Phase_C', 'Current_Phase_A', ...
                      'Current_Phase_B', 'Current_Phase_C', ...
                      'Torque', 'RPM', 'Vibration_Amplitude', ...
                      'Vibration_Frequency'});

% Standardize the new data using the loaded scaler parameters
new_data_scaled = (table2array(new_data) - mu) ./ sigma;

% Predict the fault type using the loaded SVM model
predicted_fault = predict(SVMModel, new_data_scaled);

% Display the predicted fault type
disp(['Predicted Fault Type for new data: ', char(predicted_fault)]);
