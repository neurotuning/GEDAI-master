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

function [SENSAI_score, SIGNAL_subspace_similarity, NOISE_subspace_similarity, mean_ENOVA, ENOVA_per_epoch] = SENSAI_basic(signal_data, noise_data, srate, epoch_size, refCOV, NOISE_multiplier, signal_type)

    %   Calculates the Signal & Noise Subspace Alignment Index (SENSAI) from raw EEG data
    
regularization_lambda = 0.05;
reg_val = trace(refCOV) / length(refCOV);
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(length(refCOV), 'like', refCOV);

%% Estimate Signal Quality
if nargin < 7 || isempty(signal_type)
    signal_type = 'eeg';
end

top_PCs = 3; % Always keep per-epoch subspace comparison at 3 PCs
num_chans = size(refCOV, 1);
epoch_samples = round(srate * epoch_size);

% --- Adaptive refCOV PC pool: best combination of 3 from top 5 ---
num_top_pcs_pool = min(5, num_chans);
num_pcs_to_choose = min(3, num_top_pcs_pool);
[evecs_pool, evals_pool] = eigs(refCOV_reg, num_top_pcs_pool);
[~, sidx_pool] = sort(diag(evals_pool), 'descend');
evecs_pool = evecs_pool(:, sidx_pool);
combinations = nchoosek(1:num_top_pcs_pool, num_pcs_to_choose);
num_combos = size(combinations, 1);

% --- FIX START: Truncate data to contain a whole number of epochs ---
pnts = size(signal_data, 2);
num_epochs_possible = floor(pnts / epoch_samples);
new_length = num_epochs_possible * epoch_samples;
signal_data = signal_data(:, 1:new_length);
noise_data  = noise_data(:,  1:new_length);
% --- FIX END ---

% Epoch signal and noise data
signal_EEG_epoched = reshape(signal_data, num_chans, epoch_samples, []);
noise_EEG_epoched  = reshape(noise_data,  num_chans, epoch_samples, []);
num_epochs = size(signal_EEG_epoched, 3);

% Pre-compute per-epoch signal/noise eigenvectors (independent of PC combo)
evecs_signal_all = zeros(num_chans, top_PCs, num_epochs);
evecs_noise_all  = zeros(num_chans, top_PCs, num_epochs);
ENOVA_per_epoch  = zeros(1, num_epochs);
for epoch = 1:num_epochs
    % Signal subspace
    [evS, evlS] = eig(cov(signal_EEG_epoched(:,:,epoch)'));
    [~, idxS]   = sort(diag(evlS), 'descend');
    evecs_signal_all(:,:,epoch) = evS(:, idxS(1:top_PCs));
    % Noise subspace
    [evN, evlN] = eig(cov(noise_EEG_epoched(:,:,epoch)'));
    [~, idxN]   = sort(diag(evlN), 'descend');
    evecs_noise_all(:,:,epoch)  = evN(:, idxN(1:top_PCs));
    % ENOVA
    original_epoch = signal_EEG_epoched(:,:,epoch) + noise_EEG_epoched(:,:,epoch);
    var_original   = var(original_epoch(:));
    var_noise      = var(reshape(noise_EEG_epoched(:,:,epoch), [], 1));
    if var_original > 0
        ENOVA_per_epoch(epoch) = var_noise / var_original;
    end
end
mean_ENOVA = mean(ENOVA_per_epoch);

% Search combinations for the best SENSAI score
best_SENSAI_score = -Inf;
best_evecs_Template = evecs_pool(:, 1:num_pcs_to_choose); % fallback
for c = 1:num_combos
    evecs_Template_iter = evecs_pool(:, combinations(c, :));
    sig_sim = zeros(1, num_epochs);
    noi_sim = zeros(1, num_epochs);
    for epoch = 1:num_epochs
        sig_sim(epoch) = prod(subspace_angles(evecs_signal_all(:,:,epoch), evecs_Template_iter));
        noi_sim(epoch) = prod(subspace_angles(evecs_noise_all(:,:,epoch),  evecs_Template_iter));
    end
    score_iter = 100*mean(sig_sim) - NOISE_multiplier * 100*mean(noi_sim);
    if score_iter > best_SENSAI_score
        best_SENSAI_score    = score_iter;
        best_evecs_Template  = evecs_Template_iter;
    end
end

% Compute final subspace similarities using the best PC combination
SIGNAL_subspace_similarity_distribution = zeros(1, num_epochs);
NOISE_subspace_similarity_distribution  = zeros(1, num_epochs);
for epoch = 1:num_epochs
    SIGNAL_subspace_similarity_distribution(epoch) = prod(subspace_angles(evecs_signal_all(:,:,epoch), best_evecs_Template));
    NOISE_subspace_similarity_distribution(epoch)  = prod(subspace_angles(evecs_noise_all(:,:,epoch),  best_evecs_Template));
end

SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_similarity_distribution);
NOISE_subspace_similarity  = 100 * mean(NOISE_subspace_similarity_distribution);
SENSAI_score = SIGNAL_subspace_similarity - NOISE_multiplier * NOISE_subspace_similarity;
end