function [cov_signal_epoched, cov_noise_epoched, artifact_threshold_out, Treshold1] = clean_SENSAI(artifact_threshold_in, refCOV, Eval, Evec, cov_total, signal_type, precomputed_log_prctile)
%   This GEDAI function estimates signal and noise covariances analytically
%%   Creative Commons License
%
% Copyright:  Tomas Ros & Abele Michela
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

% --- PRE-ALLOCATION ---
num_chans = size(Eval, 1);
num_epochs = size(Eval, 3);
base_diag = (1 : (num_chans + 1) : num_chans^2)';
all_indices = base_diag + (0 : num_epochs-1) * num_chans^2;
all_diagonals = Eval(all_indices(:));
% Use the magnitude (a real value) for all subsequent calculations.
magnitudes = abs(all_diagonals);
all_evals_mat = reshape(magnitudes, num_chans, num_epochs);

    %% Artifacting multiplication factor T1
    correction_factor = 1.00;
T1 = correction_factor * (105 - artifact_threshold_in) / 100;

%% Defining artifact threshold
if nargin >= 7 && ~isempty(precomputed_log_prctile)
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

%% Compute Regularized Reference Covariance
% Replicate logic from GEDAI_per_band.m to ensure we have the correct B for B-orthogonality
% refCOV argument is the raw reference covariance.
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;
regularization_lambda = 0.05;
% Using trace(refCOV)/num_chans is faster than mean(eig(refCOV)) and equivalent for SPD.
reg_val = trace(refCOV) / num_chans;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(num_chans, 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;

%% Cleaning EEG by removing outlying GEVD components
% Prepare outputs for covariances
cov_signal_epoched = zeros(num_chans, num_chans, num_epochs, 'like', Eval);
cov_noise_epoched = zeros(num_chans, num_chans, num_epochs, 'like', Eval);
threshold_val = exp(Treshold1 - 100);

for i = 1:num_epochs
    % Determine which components are artifacts based on eigenvalues
    current_evals = all_evals_mat(:, i);
    bad_indices = current_evals >= threshold_val;

    if any(bad_indices)
        VR_bad = refCOV_reg * Evec(:, bad_indices, i);
        d_bad = current_evals(bad_indices);
        cov_noise_epoched(:,:,i) = VR_bad * (d_bad .* VR_bad');
    else
        % No artifacts in this epoch
        % cov_noise_epoched(:,:,i) stays 0
    end

    % Proper Signal Covariance Estimation
    good_indices = ~bad_indices;
    
    if any(good_indices)
        VR_good = refCOV_reg * Evec(:, good_indices, i);
        d_good = current_evals(good_indices);
        cov_signal_epoched(:,:,i) = VR_good * (d_good .* VR_good');
    else
        % If no good components, signal covariance remains zero
    end

    % Enforce symmetry to allow fast symmetric eig solver downstream
    cov_noise_epoched(:,:,i) = (cov_noise_epoched(:,:,i) + cov_noise_epoched(:,:,i)') / 2;
    cov_signal_epoched(:,:,i) = (cov_signal_epoched(:,:,i) + cov_signal_epoched(:,:,i)') / 2;
end

artifact_threshold_out = artifact_threshold_in;

end
