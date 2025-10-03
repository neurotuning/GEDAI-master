% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% GNU Affero General Public License v3
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License version 3 
% as published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Affero General Public License for more details.
%
% You should have received a copy of the GNU Affero General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel)

%% Input Validation
if isempty(eeg_data)
    error('Cannot process empty data');
end
if ~ismatrix(eeg_data)
    error('Input EEG data must be a 2D matrix (channels x samples).');
end

pnts = size(eeg_data, 2);
N_EEG_electrodes = size(eeg_data, 1);

% --- MODIFIED FOR DOUBLE PRECISION ---
% Ensure input matrices are double precision to match MEX file requirements.
eeg_data = double(eeg_data);
refCOV = double(refCOV);

%% Epoch Data
% Crop data to contain a whole number of epochs
epoch_samples = srate * epoch_size;
num_epochs = floor(pnts / epoch_samples);
eeg_data = eeg_data(:, 1:epoch_samples * num_epochs);

% Epoch data stream 1
EEGdata_epoched = double(reshape(eeg_data, N_EEG_electrodes, epoch_samples, []));

% Epoch data stream 2 (shifted by half epoch)
shifting = epoch_samples / 2;
eeg_data_2 = eeg_data(:, (shifting+1):(end-shifting));
EEGdata_epoched_2 = double(reshape(eeg_data_2, N_EEG_electrodes, epoch_samples, []));
[~,~,N_epochs] = size(EEGdata_epoched);

%% Calculate Covariance Matrix per Epoch
COV = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs);
COV_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1);
for epo=1:N_epochs-1
    COV(:,:,epo) = cov(EEGdata_epoched(:,:,epo)');
    COV_2(:,:,epo) = cov(EEGdata_epoched_2(:,:,epo)');
end
COV(:,:,N_epochs) = cov(EEGdata_epoched(:,:,N_epochs)');

%% Generalized Eigendecomposition (GEVD)
regularization_lambda = 0.05;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*mean(eig(refCOV))*eye(N_EEG_electrodes);

Evec = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs);
Eval = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs);
Evec_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1);
Eval_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1);

for i=1:N_epochs-1
    [Evec(:,:,i), Eval(:,:,i)] = eig(COV(:,:,i), refCOV_reg, 'chol');
    [Evec_2(:,:,i), Eval_2(:,:,i)] = eig(COV_2(:,:,i), refCOV_reg, 'chol');
end
[Evec(:,:,N_epochs), Eval(:,:,N_epochs)] = eig(COV(:,:,N_epochs), refCOV_reg, 'chol');

%% Determine Artifact Threshold and Clean EEG
if ischar(artifact_threshold_type) && startsWith(artifact_threshold_type, 'auto')
    if strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1;
    elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3;
    else, noise_multiplier = 6; % 'auto-'
    end
    
    minThreshold = 0; maxThreshold = 12;
    
    % --- Optimization Method Switch ---
    switch optimization_type
        case 'parabolic'
            [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, EEGdata_epoched, srate, epoch_size, refCOV, Eval, Evec, noise_multiplier);
        
        case 'grid' %  Grid search functionality (can be slower)
            automatic_thresholding_step_size = 1/10;
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

[cleaned_data_1, artifacts_data_1, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec);
[cleaned_data_2, artifacts_data_2, ~] = clean_EEG(EEGdata_epoched_2, srate, epoch_size, artifact_threshold, refCOV, Eval_2, Evec_2);

%% Combine the two processed streams using cosine weighting
cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);
size_reconstructed_2 = size(cleaned_data_2, 2);
sample_end = size_reconstructed_2 - shifting;

% Apply weights to the second (shifted) stream
cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
cleaned_data_2(:, sample_end+1:end) = cleaned_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
artifacts_data_2(:, 1:shifting) = artifacts_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
artifacts_data_2(:, sample_end+1:end) = artifacts_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);

% Combine streams
cleaned_data = cleaned_data_1;
artifacts_data = artifacts_data_1;
cleaned_data(:, shifting+1:shifting+size_reconstructed_2) = cleaned_data(:, shifting+1:shifting+size_reconstructed_2) + cleaned_data_2;
artifacts_data(:, shifting+1:shifting+size_reconstructed_2) = artifacts_data(:, shifting+1:shifting+size_reconstructed_2) + artifacts_data_2;

%% Calculate final SENSAI score
[~, ~, SENSAI_score] = SENSAI(EEGdata_epoched, srate, epoch_size, artifact_threshold_out, refCOV, Eval, Evec, 1);
end
