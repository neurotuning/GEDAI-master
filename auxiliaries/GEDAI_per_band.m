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

function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, mean_ENOVA_band, ENOVA_per_epoch_band] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel, minThreshold_override, maxThreshold_override)

% Use custom minThreshold if provided, otherwise default to 0
if ~isempty(minThreshold_override)
    minThreshold = minThreshold_override;
else
    minThreshold = 0; % Default
end

% Use custom maxThreshold if provided, otherwise default to 12
if exist('maxThreshold_override', 'var') && ~isempty(maxThreshold_override)
    maxThreshold = maxThreshold_override;
else
    maxThreshold = 12; % Default
end

if isempty(eeg_data)
    error('Cannot process empty data');
end
if ~ismatrix(eeg_data)
    error('Input EEG data must be a 2D matrix (channels x samples).');
end
% pnts = size(eeg_data, 2); % Redundant
N_EEG_electrodes = size(eeg_data, 1);
% eeg_data = double(eeg_data); % REMOVED forced double cast
if ~isa(eeg_data, 'double') && ~isa(eeg_data, 'single')
    eeg_data = double(eeg_data); % Only cast if not already float
end

% Ensure refCOV matches precision of eeg_data
refCOV = cast(refCOV, 'like', eeg_data);
%% Pad and Epoch Data
pnts_original = size(eeg_data, 2); 
epoch_samples = srate * epoch_size;

remainder = rem(pnts_original, epoch_samples);
if remainder ~= 0
    samples_to_pad = epoch_samples - remainder;
    reflection_segment = eeg_data(:, end-samples_to_pad+1:end);
    padding = fliplr(reflection_segment); % Flip the segment left-to-right
    eeg_data = [eeg_data, padding];
    % disp(['Data padded with ', num2str(samples_to_pad/srate, '%.2f'), ' seconds of reflected data.']);
end

% Epoch data stream 1
EEGdata_epoched = reshape(eeg_data, N_EEG_electrodes, epoch_samples, []);

% Epoch data stream 2 (shifted by half epoch)
shifting = epoch_samples / 2; 
eeg_data_2 = eeg_data(:, (shifting+1):(end-shifting));
EEGdata_epoched_2 = reshape(eeg_data_2, N_EEG_electrodes, epoch_samples, []);
[~,~,N_epochs] = size(EEGdata_epoched);
%% Calculate Covariance Matrix per Epoch
COV = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
COV_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
for epo=1:N_epochs-1
    COV(:,:,epo) = oas_cov(EEGdata_epoched(:,:,epo)');
    COV_2(:,:,epo) = oas_cov(EEGdata_epoched_2(:,:,epo)');
end
COV(:,:,N_epochs) = oas_cov(EEGdata_epoched(:,:,N_epochs)');
%% Generalized Eigendecomposition (GEVD)
regularization_lambda = 0.05;
mu = trace(refCOV) / N_EEG_electrodes;
refCOV_reg = (1 - regularization_lambda) * refCOV + (regularization_lambda * mu) * eye(N_EEG_electrodes, 'like', refCOV);
Evec = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
Eval = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
Evec_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
Eval_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
for i=1:N_epochs-1
    [Evec(:,:,i), Eval(:,:,i)] = eig(COV(:,:,i), refCOV_reg, 'chol');
    [Evec_2(:,:,i), Eval_2(:,:,i)] = eig(COV_2(:,:,i), refCOV_reg, 'chol');
end
[Evec(:,:,N_epochs), Eval(:,:,N_epochs)] = eig(COV(:,:,N_epochs), refCOV_reg, 'chol');


%% Determine Artifact Threshold and Clean EEG
if ischar(artifact_threshold_type) && startsWith(artifact_threshold_type, 'auto')
    if strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1.5;
    elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3;
    else, noise_multiplier = 6; % 'auto-'
    end

    
    % --- Optimization Method Switch ---
    switch optimization_type
        case 'parabolic'
            [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, EEGdata_epoched, srate, epoch_size, refCOV, Eval, Evec, noise_multiplier);
        
        case 'grid' % Restored grid search functionality
            automatic_thresholding_step_size = 1/3;
            AutomaticThresholdSweep = minThreshold:automatic_thresholding_step_size:maxThreshold;
            
            SIGNAL_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
            NOISE_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
            SENSAI_score = zeros(1, length(AutomaticThresholdSweep));
            if parallel
                parfor threshold_index=1:length(AutomaticThresholdSweep)
                    artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                    % Call SENSAI function
                    [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(EEGdata_epoched, srate, epoch_size, artifact_threshold_iter, refCOV, Eval, Evec, noise_multiplier);
                end
            else
                for threshold_index=1:length(AutomaticThresholdSweep)
                    artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                    % Call SENSAI function
                    [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(EEGdata_epoched, srate, epoch_size, artifact_threshold_iter, refCOV, Eval, Evec, noise_multiplier);
                end
            end
            [~, SENSAI_index] = max(SENSAI_score);
            NOISE_changepoint_index = findchangepts(diff(smoothdata(NOISE_subspace_similarity, "movmean",6)),Statistic="mean", MaxNumChanges=2);
        
            if isempty(NOISE_changepoint_index)
                NOISE_changepoint_index = length(AutomaticThresholdSweep);      
            end
            if SENSAI_index > NOISE_changepoint_index(1)
                optimal_artifact_threshold = AutomaticThresholdSweep(NOISE_changepoint_index(1));
            else
                optimal_artifact_threshold = AutomaticThresholdSweep(SENSAI_index);
            end
    end
    
    artifact_threshold = optimal_artifact_threshold;
else
    artifact_threshold = str2double(artifact_threshold_type);
end
% Pre-calculate cosine weights for efficiency
cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);

[cleaned_data_1, artifacts_data_1, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec, cosine_weights);
[cleaned_data_2, artifacts_data_2, ~] = clean_EEG(EEGdata_epoched_2, srate, epoch_size, artifact_threshold, refCOV, Eval_2, Evec_2, cosine_weights);

% Clear Stream 2 inputs as they are no longer needed
clear EEGdata_epoched_2 Evec_2 Eval_2 COV_2;

%% Combine the two processed streams using cosine weighting
% cosine_weights is already calculated

size_reconstructed_2 = size(cleaned_data_2, 2);
sample_end = size_reconstructed_2 - shifting;
% Apply weights to the second (shifted) stream
cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
cleaned_data_2(:, sample_end+1:end) = cleaned_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
artifacts_data_2(:, 1:shifting) = artifacts_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
artifacts_data_2(:, sample_end+1:end) = artifacts_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);

% Combine streams (Optimize memory by clearing variables)
cleaned_data = cleaned_data_1;
clear cleaned_data_1; % Release memory

artifacts_data = artifacts_data_1;
clear artifacts_data_1; % Release memory

cleaned_data(:, shifting+1:shifting+size_reconstructed_2) = cleaned_data(:, shifting+1:shifting+size_reconstructed_2) + cleaned_data_2;
clear cleaned_data_2; % Release memory

artifacts_data(:, shifting+1:shifting+size_reconstructed_2) = artifacts_data(:, shifting+1:shifting+size_reconstructed_2) + artifacts_data_2;
clear artifacts_data_2; % Release memory

% Remove padding to restore original data length
cleaned_data = cleaned_data(:, 1:pnts_original);
artifacts_data = artifacts_data(:, 1:pnts_original);

%% Calculate final SENSAI score
[~, ~, SENSAI_score] = SENSAI(EEGdata_epoched, srate, epoch_size, artifact_threshold_out, refCOV, Eval, Evec, 1);

%% Calculate ENOVA (Explained Noise Variance) for this band
% Truncate data to contain whole number of epochs
pnts = size(cleaned_data, 2);
epoch_samples = srate * epoch_size;
num_epochs_possible = floor(pnts / epoch_samples);
new_length = num_epochs_possible * epoch_samples;

cleaned_data_for_enova = cleaned_data(:, 1:new_length);
artifacts_data_for_enova = artifacts_data(:, 1:new_length);

% Epoch the data
num_chans = size(cleaned_data, 1);
signal_epoched = reshape(cleaned_data_for_enova, num_chans, epoch_samples, []);
noise_epoched = reshape(artifacts_data_for_enova, num_chans, epoch_samples, []);
num_epochs = size(signal_epoched, 3);

% Calculate ENOVA per epoch
ENOVA_per_epoch_band = zeros(1, num_epochs);
for epoch = 1:num_epochs
    original_epoch = signal_epoched(:,:,epoch) + noise_epoched(:,:,epoch);
    var_original = var(original_epoch(:));
    var_noise = var(reshape(noise_epoched(:,:,epoch), [], 1));
    ENOVA_per_epoch_band(epoch) = var_noise / var_original;
end
mean_ENOVA_band = mean(ENOVA_per_epoch_band);
end