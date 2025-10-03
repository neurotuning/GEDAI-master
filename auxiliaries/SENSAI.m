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

function [SIGNAL_subspace_similarity, NOISE_subspace_similarity, SENSAI_score] = SENSAI(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec, noise_multiplier)
%   Evaluates GEDAI cleaning quality for a given threshold.
[EEGout_data, EEG_artifacts_data] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec);

%% Estimate Signal Quality
top_PCs = 3;
num_chans = size(refCOV, 1);
epoch_samples = srate * epoch_size;

% Top eigenvectors of reference covariance
[evecs_Template_cov, evals_Template_cov] = eig(refCOV);
[~, sidxS_Template_cov] = sort(diag(evals_Template_cov), 'descend');
evecs_Template_cov = evecs_Template_cov(:, sidxS_Template_cov(1:top_PCs));

% Epoch cleaned and artifact data
EEGout_epoched = reshape(EEGout_data, num_chans, epoch_samples, []);
residual_epoched = reshape(EEG_artifacts_data, num_chans, epoch_samples, []);
num_epochs = size(EEGout_epoched, 3);

SIGNAL_subspace_similarity_distribution = zeros(1, num_epochs);
NOISE_subspace_similarity_distribution = zeros(1, num_epochs);

for epoch = 1:num_epochs
    % SIGNAL SUBSPACE similarity
    cov_EEGout = cov(EEGout_epoched(:,:,epoch)');
    [evecs_EEGout, evals_EEGout] = eig(cov_EEGout);
    [~, sidxS_EEGout] = sort(diag(evals_EEGout), 'descend');
    evecs_EEGout = evecs_EEGout(:, sidxS_EEGout(1:top_PCs));
    SIGNAL_subspace_angles = subspace_angles(evecs_EEGout, evecs_Template_cov); 
    SIGNAL_subspace_similarity_distribution(epoch) = prod(cos(SIGNAL_subspace_angles));

    % NOISE SUBSPACE similarity
    cov_residual = cov(residual_epoched(:,:,epoch)');
    [evecs_residual, evals_residual] = eig(cov_residual);
    [~, sidxS_residual] = sort(diag(evals_residual), 'descend');
    evecs_residual = evecs_residual(:, sidxS_residual(1:top_PCs));
    NOISE_subspace_angles = subspace_angles(evecs_residual, evecs_Template_cov); 
    NOISE_subspace_similarity_distribution(epoch) = prod(cos(NOISE_subspace_angles));
end

%% Compute SENSAI Score
SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_similarity_distribution);
NOISE_subspace_similarity = 100 * mean(NOISE_subspace_similarity_distribution);
SENSAI_score = SIGNAL_subspace_similarity - (noise_multiplier * NOISE_subspace_similarity);
end
