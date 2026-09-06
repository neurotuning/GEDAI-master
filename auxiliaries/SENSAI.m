function [SIGNAL_subspace_similarity, NOISE_subspace_similarity, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Eval, Evec, noise_multiplier, cov_total, evecs_Template_cov, signal_type, SSI_top_PCs, precomputed_log_prctile)

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

%% Estimate Signal Quality
num_chans = size(refCOV, 1);
num_epochs = size(Eval, 3);

% Fast Subspace Iteration Setup (2-step Subspace Iteration)
M = min(size(evecs_Template_cov, 2), SSI_top_PCs);
Template_guess = evecs_Template_cov(:, 1:M);

base_diag = (1 : (num_chans + 1) : num_chans^2)';
all_indices = base_diag + (0 : num_epochs-1) * num_chans^2;
all_diagonals = Eval(all_indices(:));
magnitudes = abs(all_diagonals);
all_evals_mat = reshape(magnitudes, num_chans, num_epochs);

correction_factor = 1.00;
T1 = correction_factor * (105 - artifact_threshold) / 100;

if nargin >= 10 && ~isempty(precomputed_log_prctile)
    Treshold1 = T1 * precomputed_log_prctile;
else
    log_Eig_val_all = log(magnitudes(magnitudes > 0)) + 100;
    if strcmpi(signal_type, 'eeg')
        percentile_threshold = 98;
    elseif strcmpi(signal_type, 'meg')
        percentile_threshold = 99;
    else
        percentile_threshold = 98;
    end
    Treshold1 = T1 * prctile(log_Eig_val_all, percentile_threshold);
end
threshold_val = exp(Treshold1 - 100);

refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;
regularization_lambda = 0.05;
reg_val = trace(refCOV) / num_chans;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(num_chans, 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;

% Precompute empty subspace overlap constant (scalar across all epochs with no artifacts)
empty_subspace_sim = abs(det(Template_guess(1:M, :)));

SIGNAL_subspace_similarity_distribution = zeros(1, num_epochs);
NOISE_subspace_similarity_distribution = zeros(1, num_epochs);
tol = 1e-12;

for epoch = 1:num_epochs
    current_evals = all_evals_mat(:, epoch);
    bad_indices = current_evals >= threshold_val;
    num_bad = sum(bad_indices);
    good_indices = ~bad_indices;
    num_good = sum(good_indices);

    % --- SIGNAL SUBSPACE ---
    if num_good >= M
        % Fast associative path using direct VR projection (zero refCOV multiplications in QR loop)
        Evec_good = Evec(:, good_indices, epoch);
        VR_good = refCOV_reg * Evec_good;
        d_good = current_evals(good_indices);
        Y1_sig = VR_good * (d_good .* (VR_good' * Template_guess));
        [Q1_sig, ~] = qr(Y1_sig, 0);
        Y2_sig = VR_good * (d_good .* (VR_good' * Q1_sig));
        [evecs_signal, ~] = qr(Y2_sig, 0);
        SIGNAL_subspace_similarity_distribution(epoch) = abs(det(evecs_signal' * Template_guess));
    elseif num_good > 0
        % Fallback for rank-deficient signal
        Evec_good = Evec(:, good_indices, epoch);
        VR_good = refCOV_reg * Evec_good;
        d_good = current_evals(good_indices);
        cov_signal = VR_good * (d_good .* VR_good');
        cov_signal = (cov_signal + cov_signal') / 2;
        Y1_sig = cov_signal * Template_guess;
        [Q1_sig, ~] = qr(Y1_sig, 0);
        Y2_sig = cov_signal * Q1_sig;
        [evecs_signal, ~] = qr(Y2_sig, 0);
        SIGNAL_subspace_similarity_distribution(epoch) = abs(det(evecs_signal' * Template_guess));
    else
        % No good components
        SIGNAL_subspace_similarity_distribution(epoch) = empty_subspace_sim;
    end

    % --- NOISE SUBSPACE ---
    if strcmpi(signal_type, 'meg')
        % Energy-Fraction Rank (Optimal for MEG):
        % Uses sum(svd(Q_bad' * Template).^6) to prevent overfiltering of
        % high-power physiological rhythms that project into the leadfield.
        if num_bad > 0
            Evec_bad = Evec(:, bad_indices, epoch);
            P_bad = refCOV_reg * Evec_bad;
            [Q_bad, ~] = qr(P_bad, 0);
            s = svd(Q_bad' * Template_guess);
            NOISE_subspace_similarity_distribution(epoch) = sum(s.^6);
        else
            NOISE_subspace_similarity_distribution(epoch) = 0;
        end
    else
        % Subspace Volume Determinant (Optimal for EEG):
        % Uses 3D volume overlap so multi-component broadband artifacts
        % (EOG + EMG + NOISE) are thoroughly filtered without penalty accumulation.
        if num_bad >= M
            % Fast associative path using direct VR projection (zero refCOV multiplications in QR loop)
            Evec_bad = Evec(:, bad_indices, epoch);
            VR_bad = refCOV_reg * Evec_bad;
            d_bad = current_evals(bad_indices);
            Y1_noise = VR_bad * (d_bad .* (VR_bad' * Template_guess));
            [Q1_noise, ~] = qr(Y1_noise, 0);
            Y2_noise = VR_bad * (d_bad .* (VR_bad' * Q1_noise));
            [evecs_noise, ~] = qr(Y2_noise, 0);
            NOISE_subspace_similarity_distribution(epoch) = abs(det(evecs_noise' * Template_guess));
        elseif num_bad > 0
            % Fallback for rank-deficient noise
            Evec_bad = Evec(:, bad_indices, epoch);
            VR_bad = refCOV_reg * Evec_bad;
            d_bad = current_evals(bad_indices);
            cov_noise = VR_bad * (d_bad .* VR_bad');
            cov_noise = (cov_noise + cov_noise') / 2;
            
            if max(abs(cov_noise(:))) < tol
                Y1_noise = eye(num_chans, M);
            else
                Y1_noise = cov_noise * Template_guess;
            end
            [Q1_noise, ~] = qr(Y1_noise, 0);
            Y2_noise = cov_noise * Q1_noise;
            [evecs_noise, ~] = qr(Y2_noise, 0);
            NOISE_subspace_similarity_distribution(epoch) = abs(det(evecs_noise' * Template_guess));
        else
            % No bad components
            NOISE_subspace_similarity_distribution(epoch) = empty_subspace_sim;
        end
    end
end

%% Compute SENSAI Score
SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_similarity_distribution);
NOISE_subspace_similarity = 100 * mean(NOISE_subspace_similarity_distribution);
SENSAI_score = SIGNAL_subspace_similarity - (noise_multiplier * NOISE_subspace_similarity);
end