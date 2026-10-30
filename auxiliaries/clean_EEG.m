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

function [cleaned_data, artifacts_data, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold_in, refCOV, Eval, Evec, cosine_weights, signal_type, global_start_epoch_idx, global_total_epochs, ~)
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
% eigenvalues (all channels x all epochs) and scaling per-epoch by T1_array.
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
epoch_samples = size(EEGdata_epoched, 2);
artifacts = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
cleaned_epoched_data = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
if nargin < 8 || isempty(cosine_weights)
    cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
end
half_epoch = floor(epoch_samples / 2);

% --- Purely Spatial Dipolarity-Inflated Threshold & Subspace Shield (EEG) ---
% Frequency-agnostic spatial protection for dipolar cortical rhythms (delta,
% theta, alpha, mu, beta) whose GEVD eigenvalues modestly exceed the baseline
% threshold due to leadfield eigenvalue roll-off across modes 3..K9.
enable_spatial_shield = strcmpi(signal_type, 'eeg') && (num_chans >= 8);
if enable_spatial_shield
    [Vs_lf, Ds_lf] = eig(double(refCOV_reg));
    [~, ord_lf] = sort(real(diag(Ds_lf)), 'descend');
    V_lf = real(Vs_lf(:, ord_lf));
    K6 = min(8, min(max(3, round(0.22 * num_chans)), max(1, num_chans - 2)));
    K9 = min(11, min(max(5, round(0.33 * num_chans)), max(1, num_chans - 2)));
    V6 = cast(V_lf(:, 1:K6), 'like', refCOV_reg);
    V9 = cast(V_lf(:, 1:K9), 'like', refCOV_reg);
    win_len = max(5, round(0.20 * epoch_samples));
end

for i = 1:num_epochs
    ev_i = abs(diag(Eval(:,:,i)));
    thr_i = exp(Treshold1_array(i) - 100);
    bad_cols = find(ev_i >= thr_i);
    
    if ~isempty(bad_cols)
        W_bad = Evec(:, bad_cols, i);
        tc_bad = W_bad' * EEGdata_epoched(:, :, i);
        P_bad = refCOV_reg * W_bad;
        
        if enable_spatial_shield
            ep_med_ev = median(ev_i(ev_i > 0));
            for k = 1:numel(bad_cols)
                c_idx = bad_cols(k);
                ev_ep_r = ev_i(c_idx) / max(ep_med_ev, 1e-12);
                % Within-epoch eigenvalue gate: genuine dipolar cortical rhythms have eigenvalues within
                % ~10x-24x the epoch's median GEVD eigenvalue (determined by the 5% leadfield regularization
                % floor), whereas high-contrast artifacts exceed 32x the epoch median.
                g_ep = max(0, min(1.0, (32.0 - ev_ep_r) / 8.0));
                
                if g_ep > 0
                    p_raw = P_bad(:, k);
                    p_unit = p_raw / max(norm(p_raw), 1e-12);
                    w_unit = W_bad(:, k) / max(norm(W_bad(:, k)), 1e-12);
                    
                    % Spatial smoothness (fraction of forward pattern energy in K6 and K9 leadfield harmonics)
                    sm6 = sum((V6' * p_unit).^2);
                    sm9 = sum((V9' * p_unit).^2);
                    pw_cos = abs(p_unit' * w_unit);
                    ev_r = ev_i(c_idx) / max(thr_i, 1e-12);
                    
                    % Hinged Dipolarity-Inflated Threshold:
                    s6 = max(0, (sm6 - 0.60) / 0.40);
                    s9 = max(0, (sm9 - 0.70) / 0.30);
                    g_pw = max(0.20, min(1.0, (0.60 - pw_cos) / 0.20));
                    eff_r = 1.0 + 3.0 * (0.5 * s6 + 0.5 * s9) * g_pw * g_ep;
                    
                    if ev_r < eff_r
                        tc_bad(k, :) = 0;
                    else
                        if sm9 > 0.70 && epoch_samples >= 16
                            loc_p = movmean(tc_bad(k, :).^2, win_len);
                            thr_p = (1.0 + (eff_r - 1.0) * g_ep) * thr_i;
                            w_t = max(0, 1 - (thr_p ./ max(loc_p, 1e-12)).^2);
                            tc_bad(k, :) = w_t .* tc_bad(k, :);
                        end
                        if ev_r < (1.0 + 2.0 * g_ep) && sm6 > 0.40
                            s6_sub = min(1.0, max(0, (sm6 - 0.40) / 0.25));
                            keep_w = g_ep * s6_sub * max(0, 1 - ((ev_r - 1.0) / max(2.0 * g_ep, 1e-6))^2);
                            P_bad(:, k) = p_raw - keep_w * (V6 * (V6' * p_raw));
                        end
                    end
                end
            end
        end
        
        Signal_to_remove = P_bad * tc_bad;
    else
        Signal_to_remove = zeros(num_chans, epoch_samples, 'like', EEGdata_epoched);
    end
    
    artifacts(:, :, i) = Signal_to_remove;
    cleaned_epoch = EEGdata_epoched(:,:,i) - Signal_to_remove;
    
    % Apply cosine windowing to mitigate edge effects from epoching
    global_idx = global_start_epoch_idx + i - 1;
    if global_idx == 1
        cleaned_epoch(:, half_epoch+1:end) = cleaned_epoch(:, half_epoch+1:end) .* cosine_weights(:, half_epoch+1:end);
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
