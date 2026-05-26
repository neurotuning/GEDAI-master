function [optimalThreshold, maxSENSAIScore] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs)

max_number_of_epochs = 500; % if EEG recording is long (default = 500 epochs)
number_of_epochs = size(COV, 3);

if  number_of_epochs > max_number_of_epochs
stream = RandStream('twister', 'Seed', 2);
random_epochs = randperm(stream, number_of_epochs, max_number_of_epochs);
% EEGdata_epoched removed
Eval = Eval (:, :, random_epochs);
Evec = Evec (:, :,random_epochs);
% Also subset COV to match
COV = COV(:,:,random_epochs);

else
end

sensaifunc = @(artifactThreshold) SENSAIObjective(artifactThreshold, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov,signal_type, SSI_top_PCs);
[optimalThreshold, negMaxSENSAIScore] = local_fminbnd(sensaifunc, minThreshold, maxThreshold, 1e-2);

% % 1. Define the optimization variable (the threshold)
% vars = optimizableVariable('threshold', [minThreshold, maxThreshold]);
% 
% % 2. Define the objective function
% % bayesopt expects a function that takes a TABLE and returns the objective value
% objFcn = @(tbl) sensaifunc(tbl.threshold);
% 
% % 3. Run the Bayesian Optimization
% results = bayesopt(objFcn, vars, ...
%     'MaxObjectiveEvaluations', 50, ... % Number of iterations
%     'NumSeedPoints', 10, ...            % Initial random points
%     'PlotFcn', [], ...                 % Set to {} to see progress plots
%     'UseParallel', false, ...
%     'Verbose', 0);                     % Set to 1 to see logs
% 
% % 4. Extract the results to match your original variables
% optimalThreshold = results.XAtMinObjective.threshold;
% negMaxSENSAIScore = results.MinObjective;


    function objective = SENSAIObjective(artifact_threshold, refCOV, Eval, Evec, noise_multiplier_obj, cov_total, evecs_Template_cov_obj,signal_type, SSI_top_PCs)
        % Compute the negative SENSAI score for the objective function
        [~, ~, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Eval, Evec, noise_multiplier_obj, cov_total, evecs_Template_cov_obj, signal_type, SSI_top_PCs);
        objective = -SENSAI_score;
    end
end