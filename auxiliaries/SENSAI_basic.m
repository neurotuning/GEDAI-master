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

function [SENSAI_score, SIGNAL_subspace_similarity, NOISE_subspace_similarity] = SENSAI_basic(signal_data, noise_data, srate, epoch_size, refCOV, NOISE_multiplier)

%% Regularize reference covariance matrix
regularization_lambda = 0.05;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*mean(eig(refCOV))*eye(length(refCOV));

%% Estimate Signal Quality
top_PCs = 3;
num_chans = size(refCOV, 1);
epoch_samples = srate * epoch_size;

% Top eigenvectors of refCOV subspace
[evecs_Template_cov, evals_Template_cov] = eig(refCOV_reg);
[~, sidxS_Template_cov] = sort(diag(evals_Template_cov), 'descend');
evecs_Template_cov = evecs_Template_cov(:, sidxS_Template_cov(1:top_PCs));

% Epoch signal and noise data
signal_EEG_epoched = reshape(signal_data, num_chans, epoch_samples, []);
noise_EEG_epoched = reshape(noise_data, num_chans, epoch_samples, []);
num_epochs = size(signal_EEG_epoched, 3);

SIGNAL_subspace_similarity_distribution = zeros(1, num_epochs);
NOISE_subspace_similarity_distribution = zeros(1, num_epochs);

for epoch = 1:num_epochs
    % SIGNAL SUBSPACE
    cov_signal_EEG = cov(signal_EEG_epoched(:,:,epoch)');
    [evecs_signal_EEG, evals_signal_EEG] = eig(cov_signal_EEG);
    [~, sidxS_signal_EEG] = sort(diag(evals_signal_EEG), 'descend');
    evecs_signal_EEG = evecs_signal_EEG(:, sidxS_signal_EEG(1:top_PCs));
    SIGNAL_subspace_angles = subspace_angles(evecs_signal_EEG, evecs_Template_cov); 
    SIGNAL_subspace_similarity_distribution(epoch) = prod(cos(SIGNAL_subspace_angles));

    % NOISE SUBSPACE
    cov_noise = cov(noise_EEG_epoched(:,:,epoch)');
    [evecs_noise, evals_noise] = eig(cov_noise);
    [~, sidxS_noise] = sort(diag(evals_noise), 'descend');
    evecs_noise = evecs_noise(:, sidxS_noise(1:top_PCs));
    NOISE_subspace_angles = subspace_angles(evecs_noise, evecs_Template_cov); 
    NOISE_subspace_similarity_distribution(epoch) = prod(cos(NOISE_subspace_angles));
end

SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_similarity_distribution);
NOISE_subspace_similarity = 100 * mean(NOISE_subspace_similarity_distribution);
SENSAI_score = SIGNAL_subspace_similarity - NOISE_multiplier * NOISE_subspace_similarity;
end
