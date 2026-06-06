function [SIGNAL_subspace_similarity, NOISE_subspace_similarity, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Eval, Evec, noise_multiplier, cov_total, evecs_Template_cov, signal_type, SSI_top_PCs)

                       %   Evaluates GEDAI cleaning quality for a given threshold.
%%   Creative Commons License
%
%   Credits:  Tomas Ros & Abele Michela 
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its CONTRIBUTORS
% may be used to endorse or promote products derived from this software without
% specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
% THE POSSIBILITY OF SUCH DAMAGE.

[cov_signal_epoched, cov_noise_epoched] = clean_SENSAI(artifact_threshold, refCOV, Eval, Evec, cov_total, signal_type);

%% Estimate Signal Quality
%% Estimate Signal Quality
num_chans = size(refCOV, 1);

% Top eigenvectors of reference covariance (Calculated outside and passed in)

num_epochs = size(cov_signal_epoched, 3);

SIGNAL_subspace_similarity_distribution = zeros(1, num_epochs);
NOISE_subspace_similarity_distribution = zeros(1, num_epochs);
baseline_similarity_distribution = zeros(1, num_epochs);

% Fast Subspace Iteration Setup (2-step Subspace Iteration)
M = min(size(evecs_Template_cov, 2), SSI_top_PCs);
Template_guess = evecs_Template_cov(:, 1:M);

for epoch = 1:num_epochs
    % SIGNAL SUBSPACE similarity via 2-step Fast Subspace Iteration
    cov_signal = cov_signal_epoched(:,:,epoch);
    Y1_sig = cov_signal * Template_guess;
    [Q1_sig, ~] = qr(Y1_sig, 0);
    Y2_sig = cov_signal * Q1_sig;
    [evecs_signal, ~] = qr(Y2_sig, 0);
    SIGNAL_subspace_similarity_distribution(epoch) = prod(subspace_angles(evecs_signal, evecs_Template_cov));

    % % NOISE SUBSPACE similarity via 2-step Fast Subspace Iteration
    cov_noise = cov_noise_epoched(:,:,epoch);
    
    % % Define a tolerance (e.g., 1e-12 or using eps)
    tol = 1e-12; 
    % Check if the maximum absolute value is below the tolerance
    if max(abs(cov_noise(:))) < tol
        % Use the size of Template_guess to construct the identity-like matrix
        Y1_noise = eye(size(Template_guess)); 
    else
        Y1_noise = cov_noise * Template_guess;
    end

    [Q1_noise, ~] = qr(Y1_noise, 0);
    Y2_noise = cov_noise * Q1_noise;
    [evecs_noise, ~] = qr(Y2_noise, 0);
    NOISE_subspace_similarity_distribution(epoch) = prod(subspace_angles(evecs_noise, evecs_Template_cov));

    % BASELINE SUBSPACE similarity (prior to any denoising)
    cov_raw = cov_total(:,:,epoch);
    Y1_raw = cov_raw * Template_guess;
    [Q1_raw, ~] = qr(Y1_raw, 0);
    Y2_raw = cov_raw * Q1_raw;
    [evecs_raw, ~] = qr(Y2_raw, 0);
    baseline_similarity_distribution(epoch) = prod(subspace_angles(evecs_raw, evecs_Template_cov));
end

%% Compute Weighted SENSAI Score (Graded Rank-Based Weighting)
[~, sort_idx] = sort(baseline_similarity_distribution, 'ascend');
raw_weights = zeros(1, num_epochs);
raw_weights(sort_idx) = num_epochs:-1:1;

sum_weights = sum(raw_weights);
if sum_weights > 0
    weights = raw_weights / sum_weights;
else
    weights = ones(1, num_epochs) / num_epochs;
end

SIGNAL_subspace_similarity = 100 * sum(SIGNAL_subspace_similarity_distribution .* weights);
NOISE_subspace_similarity = 100 * sum(NOISE_subspace_similarity_distribution .* weights);
SENSAI_score = SIGNAL_subspace_similarity - (noise_multiplier * NOISE_subspace_similarity);
end