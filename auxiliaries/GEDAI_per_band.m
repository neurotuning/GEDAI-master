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

function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel, signal_type, minThreshold)

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

% Default signal_type if not provided
if nargin < 9 || isempty(signal_type)
    signal_type = 'eeg'; 
end

% Default minThreshold if not provided
if nargin < 10 || isempty(minThreshold)
    minThreshold = 0;
end

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
    COV(:,:,epo) = cov(EEGdata_epoched(:,:,epo)');
    COV_2(:,:,epo) = cov(EEGdata_epoched_2(:,:,epo)');
end
COV(:,:,N_epochs) = cov(EEGdata_epoched(:,:,N_epochs)');
%% Generalized Eigendecomposition (GEVD)
regularization_lambda = 0.05;
reg_val = trace(refCOV) / N_EEG_electrodes;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(N_EEG_electrodes, 'like', refCOV);
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
%% Determine Noise Multiplier and Optimization Parameters
if ischar(artifact_threshold_type) && startsWith(artifact_threshold_type, 'auto')
    if strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1.5;
    elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3; 
    elseif strcmp(artifact_threshold_type,'auto-'), noise_multiplier = 6;
    else, noise_multiplier = 3; 
    end
else
    % Numeric input defines noise_multiplier linkage
    if isnumeric(artifact_threshold_type)
        val = artifact_threshold_type;
    else
        val = str2double(artifact_threshold_type);
    end
    noise_multiplier = 10 - val;
end

% Ensure valid multiplier
if isnan(noise_multiplier), noise_multiplier = 3; end

% --- Run SENSAI Optimization ---

% Pre-calculate RefCOV eigenvectors for SENSAI

if strcmpi(signal_type, 'eeg')
    refCOV_top_PCs=3;
    disp(['EEG  refCOV PCs: ' num2str(refCOV_top_PCs)]);
    SSI_top_PCs = 3;
    disp(['EEG  SSI PCs: ' num2str(SSI_top_PCs) newline]);

elseif strcmpi(signal_type, 'meg')
    % Adaptive: minimum PCs explaining >= 70% of refCOV variance
    % Use refCOV_reg (regularized, always well-conditioned)
    % refCOV_reg=refCOV;
    all_evals_refCOV = eig(refCOV_reg);
    all_evals_refCOV = sort(all_evals_refCOV, 'descend');
    cumvar_refCOV = cumsum(all_evals_refCOV) / sum(all_evals_refCOV);
    refCOV_top_PCs = find(cumvar_refCOV >= 0.85, 1, 'first');
    refCOV_top_PCs = max(1, min(refCOV_top_PCs, N_EEG_electrodes - 1));
    fprintf('MEG  RefCOV PCs: %d (%.0f%% var)\n', refCOV_top_PCs, 100 * cumvar_refCOV(refCOV_top_PCs));


    % Top PCs for SENSAI (separate from refCOV top PCs)
    refCOV_ratio=(refCOV_top_PCs/(noise_multiplier + 3))^1/2;
    SSI_top_PCs = round(refCOV_ratio + 3);
    disp(['MEG  SSI PCs: ' num2str(SSI_top_PCs) newline]);
end

    % Use eigs for truncated decomposition (efficiency optimization)
    [evecs_Template_cov, evals_Template_cov] = eigs(refCOV_reg, refCOV_top_PCs);

    % Ensure sorted order
    [~, sidxS_Template_cov] = sort(diag(evals_Template_cov), 'descend');
    evecs_Template_cov = evecs_Template_cov(:, sidxS_Template_cov);


maxThreshold = 12;

% --- Optimization Method Switch ---
switch optimization_type
    case 'parabolic'
        [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs);
    
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
                [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs);
            end
        else
            for threshold_index=1:length(AutomaticThresholdSweep)
                artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                % Call SENSAI function
                [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs);
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
% Pre-calculate cosine weights for efficiency
cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);

[cleaned_data_1, artifacts_data_1, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec, cosine_weights, signal_type);
[cleaned_data_2, artifacts_data_2, ~] = clean_EEG(EEGdata_epoched_2, srate, epoch_size, artifact_threshold, refCOV, Eval_2, Evec_2, cosine_weights, signal_type);

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
[~, ~, SENSAI_score] = SENSAI(artifact_threshold_out, refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs);

% Calculate mean ENOVA for this band (average of per-epoch variance ratios)
original_data = cleaned_data + artifacts_data;

% Reshape into epochs (channels x samples x epochs)
epoch_samples = srate * epoch_size;
% Handle potential padding/truncation: use floor to get full epochs
num_epochs_possible = floor(size(original_data, 2) / epoch_samples);
len_to_use = num_epochs_possible * epoch_samples;

original_epoched = reshape(original_data(:, 1:len_to_use), size(original_data, 1), epoch_samples, []);
artifacts_epoched = reshape(artifacts_data(:, 1:len_to_use), size(artifacts_data, 1), epoch_samples, []);

num_epochs = size(original_epoched, 3);
enova_per_epoch = zeros(1, num_epochs);

for i = 1:num_epochs
    % Calculate variance for this epoch (across all channels and time points in epoch)
    var_orig = var(reshape(original_epoched(:,:,i), [], 1));
    var_art = var(reshape(artifacts_epoched(:,:,i), [], 1));
    
    if var_orig > 0
        enova_per_epoch(i) = var_art / var_orig;
    else
        enova_per_epoch(i) = 0; % Avoid division by zero
    end
end

if num_epochs > 0
    ENOVA = mean(enova_per_epoch);
else
    ENOVA = 0;
end
end