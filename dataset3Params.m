function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% initializing with temporary values; these are not used
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% we take 8 samples of C values and 8 samples of sigma_squared values giving
% 64 different combinations (C,s). For each of these 64 combinations we train
% the SVM model on X and y.

C_list = [0.01 0.03 0.1 0.3 1 3 10 30]; % sample C values
s_list = [0.01 0.03 0.1 0.3 1 3 10 30]; % sample sigma_squared values

% array to store computed cost/errors and corresponding C and s values
% each row i of the form [C_list(i) s_list(i) error_list(i)] 
% this array has 8*8 i.e. 64 rows since there are 64 combinations of C and s to
% to be tried
cLength = length(C_list);
sLength = length(s_list);
error_list = zeros((cLength * sLength), 3);

% temporary variables
cTemp = 0;
sTemp = 0;
errorTemp = 0;

% for indexing across error_list
row = 1;

% train on X and evaluate on Xval for all combinations of C and sigma_squared
% save list of errors from the evaluation
for i = 1:cLength
    
    cTemp = C_list(i);  
    
    for j = 1:sLength
      
      sTemp = s_list(j);
      
      % train a SVM model using traing set (X, y), C value ctemp and 
      % gaussian kernel with sigma square value stemp
      fprintf('\nTraining SVM with RBF Kernel for model %0.0f \n', row);
      model= svmTrain(X, y, cTemp, @(x1, x2) gaussianKernel(x1, x2, sTemp)); 
      
      % use above model to make predictions on cross validation set (Xval, yval)
      % return value pred is a vector of 0s and 1s corresponding to hTheta
      pred = svmPredict(model, Xval);
      
      % compare pred with yval; more examples where pred == y, implies high
      % accuracy. For classification problems, Error (or cost) is fraction of 
      % predictions (i.e hThetax) that don't match labels yval 
      errorTemp = mean(double(pred ~= yval)) * 100;
      % alternately Error (or cost) percetange is 100 - accuracy percentage.
      % errorTemp = 100 - mean(double(pred == yval)) * 100;
      error_list(row,:) = [cTemp sTemp errorTemp];

      % increment error_list index
      row = row + 1;
    endfor
  
endfor
% after the above loops we have the the error_list vector filled with C and
% sigma_squared combinations and corresponding errors when evaluated on Xval 

% compute which row of error_list has the minimum error;
% return cTemp and sTemp in that row.
[minValue, minIndex] = min(error_list(:,3));
C = error_list(minIndex, 1);
sigma = error_list(minIndex, 2);

% =========================================================================
end
