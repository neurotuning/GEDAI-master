% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [cleaned_data, artifacts_data, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold_in, refCOV, Eval, Evec, cosine_weights, signal_type, global_start_epoch_idx, global_total_epochs)
%   This GEDAI function reconstructs the signal after removing artifactual components

% --- PRE-ALLOCATION ---
num_chans = size(Eval, 1);
num_epochs = size(Eval, 3);

if nargin < 10 || isempty(global_start_epoch_idx)
    global_start_epoch_idx = 1;
end
if nargin < 11 || isempty(global_total_epochs)
    global_total_epochs = num_epochs;
end
% Pre-allocate the array 
all_diagonals = zeros(num_chans * num_epochs, 1, 'like', Eval);
for i = 1:num_epochs
    start_idx = (i-1) * num_chans + 1;
    end_idx = i * num_chans;
    all_diagonals(start_idx:end_idx) = diag(Eval(:,:,i));
end
% Use the magnitude (a real value) for all subsequent calculations.
magnitudes = abs(all_diagonals);
all_evals_mat = reshape(magnitudes, num_chans, num_epochs);

if isscalar(artifact_threshold_in)
    artifact_threshold_in = repmat(artifact_threshold_in, 1, num_epochs);
end

%% Parameter-free Left-Half Gaussian Mirror + Right-Tail Kurtosis cutoff (OptA)
evals_sorted = sort(all_evals_mat, 1, 'ascend');
if num_chans > 1
    active_evals = evals_sorted(2:end, :);
else
    active_evals = evals_sorted;
end
pos_evals = active_evals(active_evals > 0);
if ~isempty(pos_evals)
    y_log = log(pos_evals);
    mu0 = median(y_log);
    left_dev = mu0 - y_log(y_log <= mu0);
    sig_mad = 1.4826 * median(left_dev);
    sig_rms = sqrt(mean(left_dev.^2));
    sig0 = 0.5 * (sig_mad + sig_rms);
    right_dev = y_log(y_log >= mu0) - mu0;
    sig_tail4 = (mean(right_dev.^4))^(1/4);
    y_anchor = mu0 + 2.50 * sig0 + 0.85 * (sig_tail4 - sig0);
    scale_step = (y_anchor - (mu0 + 0.25 * sig0)) / 7.0;
    base_log_cut = y_anchor + 5.0 * scale_step;
    Treshold1_array = base_log_cut - artifact_threshold_in * scale_step + 100;
else
    Treshold1_array = inf(1, num_epochs);
end


%% Compute Regularized Reference Covariance for fast linear system bypass
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;
regularization_lambda = 0.05;
reg_val = trace(refCOV) / num_chans;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(num_chans, 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;

%% Cleaning EEG by removing outlying GEVD components
epoch_samples = round(srate * epoch_size);
artifacts = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
cleaned_epoched_data = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
if nargin < 8 || isempty(cosine_weights)
    cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
end
half_epoch = epoch_samples/2;

for i = 1:num_epochs
    component_spatial_filter = Evec(:,:,i);
    
    % --- OPTIMIZATION START ---
    % 1. Create a logical mask of indices to zero out
    % (We zero out the SIGNAL components to reconstruct the NOISE to subtract)
    signal_indices = abs(diag(Eval(:,:,i))) < exp(Treshold1_array(i) - 100);
    
    % 2. Apply the mask (Vectorized)
    component_spatial_filter(:, signal_indices) = 0;
    % --- OPTIMIZATION END ---

    artifacts_timecourses = component_spatial_filter' * EEGdata_epoched(:,:,i);    
    Signal_to_remove = refCOV_reg * (Evec(:,:,i) * artifacts_timecourses);
    
    artifacts(:, :, i) = Signal_to_remove;
    cleaned_epoch = EEGdata_epoched(:,:,i) - Signal_to_remove;
    
    % Apply cosine windowing to mitigate edge effects from epoching
    global_idx = global_start_epoch_idx + i - 1;
    if global_idx == 1
        cleaned_epoch(:, half_epoch+1:end) = cleaned_epoch(:, half_epoch+1:end) .* cosine_weights(:, half_epoch+1:end);
        artifacts(:, :, i) = artifacts(:, :, i); % Copy first
        artifacts(:, half_epoch+1:end, i) = artifacts(:, half_epoch+1:end, i) .* cosine_weights(:, half_epoch+1:end);
    elseif global_idx == global_total_epochs
        cleaned_epoch(:, 1:half_epoch) = cleaned_epoch(:, 1:half_epoch) .* cosine_weights(:, 1:half_epoch);
        artifacts(:, 1:half_epoch, i) = artifacts(:, 1:half_epoch, i) .* cosine_weights(:, 1:half_epoch);
    else
        cleaned_epoch = cleaned_epoch .* cosine_weights;
        artifacts(:, :, i) = artifacts(:, :, i) .* cosine_weights;
    end
    
    cleaned_epoched_data(:,:,i) = cleaned_epoch;
end

% Reshape data back to continuous form and return outputs
cleaned_data = reshape(cleaned_epoched_data, num_chans, []);
artifacts_data = reshape(artifacts, num_chans, []);
artifact_threshold_out = artifact_threshold_in;

end