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
log_Eig_val_all = log(magnitudes(magnitudes > 0)) + 100;


%% Artifacting multiplication factor T1
correction_factor = 1.00;

if isscalar(artifact_threshold_in)
    artifact_threshold_in = repmat(artifact_threshold_in, 1, num_epochs);
end

T1_array = correction_factor * (105 - artifact_threshold_in) / 100;

%% Defining artifact threshold

    if strcmpi(signal_type, 'eeg')
       percentile_threshold = 98;
      
    elseif strcmpi(signal_type, 'meg')
           percentile_threshold = 99;
    end

% Compute Treshold1 per epoch to match exactly how clean_SENSAI evaluates it:
% SENSAI uses the GLOBAL percentile of ALL eigenvalues in the window it was
% optimized on. We replicate that here by computing one percentile over ALL
% eigenvalues (all channels × all epochs) and scaling per-epoch by T1_array.
global_log_prctile = prctile(log_Eig_val_all, percentile_threshold);
Treshold1_array = T1_array * global_log_prctile;


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

% Precompute top-K leadfield dipolar projector P_K and alpha-band FFT bins
% to protect dipolar physiological rhythms from high-frequency tail leakage.
use_dipolar_gate = strcmpi(signal_type, 'eeg') && (epoch_samples >= round(0.45 * srate));
if use_dipolar_gate
    [V_lf, D_lf] = eig(refCOV_reg);
    [~, ord_lf] = sort(real(diag(D_lf)), 'descend');
    K_dip = min(6, max(2, floor(num_chans / 2)));
    V_K = real(V_lf(:, ord_lf(1:K_dip)));
    P_orth_K = eye(num_chans, 'like', refCOV_reg) - V_K * V_K';
    freqs_fft = (0:(epoch_samples - 1)) * (srate / epoch_samples);
    alpha_bins = (freqs_fft >= 7 & freqs_fft <= 14);
    nyq_bins = (freqs_fft > 0 & freqs_fft <= (srate / 2));
end

for i = 1:num_epochs
    component_spatial_filter = Evec(:,:,i);
    
    % --- OPTIMIZATION START ---
    % 1. Create a logical mask of indices to zero out
    % (We zero out the SIGNAL components to reconstruct the NOISE to subtract)
    signal_indices = abs(diag(Eval(:,:,i))) < exp(Treshold1_array(i) - 100);
    
    % 2. Apply the mask (Vectorized)
    component_spatial_filter(:, signal_indices) = 0;
    % --- OPTIMIZATION END ---

    if use_dipolar_gate && any(~signal_indices)
        bad_idx = find(~signal_indices);
        Evec_bad = Evec(:, bad_idx, i);
        A_bad = refCOV_reg * Evec_bad;
        tc_bad = Evec_bad' * EEGdata_epoched(:,:,i);
        F_bad = abs(fft(double(tc_bad), [], 2)).^2;
        alpha_frac = sum(F_bad(:, alpha_bins), 2) ./ max(sum(F_bad(:, nyq_bins), 2), 1e-12);
        for b = 1:numel(bad_idx)
            a_b = A_bad(:, b);
            lf_align = sum((V_K' * a_b).^2) / max(sum(a_b.^2), 1e-12);
            if lf_align >= 0.65 && alpha_frac(b) >= 0.25
                A_bad(:, b) = P_orth_K * a_b;
            end
        end
        Signal_to_remove = A_bad * tc_bad;
    else
        artifacts_timecourses = component_spatial_filter' * EEGdata_epoched(:,:,i);    
        Signal_to_remove = refCOV_reg * (Evec(:,:,i) * artifacts_timecourses);
    end
    
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