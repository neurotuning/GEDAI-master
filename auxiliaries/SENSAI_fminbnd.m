function [optimalThreshold, maxSENSAIScore] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs)

max_number_of_epochs = 500; % if EEG recording is long (default = 500 epochs)
number_of_epochs = size(Eval, 3);

if number_of_epochs > max_number_of_epochs
    stream = RandStream('twister', 'Seed', 2);
    random_epochs = randperm(stream, number_of_epochs, max_number_of_epochs);
    Eval = Eval(:, :, random_epochs);
    Evec = Evec(:, :, random_epochs);
end

% Precompute log-percentile of eigenvalues once for all optimization iterations
num_chans = size(refCOV, 1);
num_epochs_sub = size(Eval, 3);
base_diag = (1 : (num_chans + 1) : num_chans^2)';
all_indices = base_diag + (0 : num_epochs_sub-1) * num_chans^2;
magnitudes = abs(Eval(all_indices(:)));
log_Eig_val_all = log(magnitudes(magnitudes > 0)) + 100;

if strcmpi(signal_type, 'eeg')
    percentile_threshold = 98;
elseif strcmpi(signal_type, 'meg')
    percentile_threshold = 99;
else
    percentile_threshold = 98;
end
precomputed_log_prctile = prctile(log_Eig_val_all, percentile_threshold);

sensaifunc = @(artifactThreshold) SENSAIObjective(artifactThreshold, refCOV, Eval, Evec, noise_multiplier, [], evecs_Template_cov, signal_type, SSI_top_PCs, precomputed_log_prctile);
[optimalThreshold, negMaxSENSAIScore] = local_fminbnd(sensaifunc, minThreshold, maxThreshold, 1e-2);
maxSENSAIScore = -negMaxSENSAIScore;

    function objective = SENSAIObjective(artifact_threshold, refCOV_obj, Eval_obj, Evec_obj, noise_multiplier_obj, cov_total, evecs_Template_cov_obj, signal_type_obj, SSI_top_PCs_obj, precomputed_log_prctile_obj)
        % Compute the negative SENSAI score for the objective function
        [~, ~, SENSAI_score] = SENSAI(artifact_threshold, refCOV_obj, Eval_obj, Evec_obj, noise_multiplier_obj, cov_total, evecs_Template_cov_obj, signal_type_obj, SSI_top_PCs_obj, precomputed_log_prctile_obj);
        objective = -SENSAI_score;
    end
end