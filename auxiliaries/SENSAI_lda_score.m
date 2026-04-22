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

function lda_accuracy = SENSAI_lda_score(signal_data, artifact_data, refCOV, srate, epoch_size, signal_type)
% SENSAI_LDA_SCORE  Compute 2D LDA separability between signal and noise epochs.
%
%   Extracts per-epoch [SSI, log-power] features for cleaned signal and
%   artifact data, then runs a 5-fold cross-validated Linear Discriminant
%   Analysis.  Returns classification accuracy in [0, 100] %.
%
%   A higher score means signal and artifact epochs are more separable in
%   the SSI × power space → better denoising.
%
% Inputs:
%   signal_data   – channels × samples matrix of cleaned EEG
%   artifact_data – channels × samples matrix of removed artifacts
%   refCOV        – channels × channels leadfield covariance (reference)
%   srate         – sampling rate (Hz)
%   epoch_size    – epoch duration (seconds)
%   signal_type   – 'eeg' or 'meg' (controls number of top PCs for SSI)
%
% Output:
%   lda_accuracy  – scalar in [0, 100]; NaN if LDA fails

%% ── 0. Defaults ─────────────────────────────────────────────────────────
if nargin < 6 || isempty(signal_type)
    signal_type = 'eeg';
end

%% ── 1. Number of top PCs for SSI ────────────────────────────────────────
if strcmpi(signal_type, 'meg') || size(refCOV, 1) <= 100
    SSI_top_PCs = 4;
else
    SSI_top_PCs = 3;
end

%% ── 2. Reference basis (top PCs of refCOV) ──────────────────────────────
regularization_lambda = 0.05;
reg_val    = trace(refCOV) / size(refCOV, 1);
refCOV_reg = (1 - regularization_lambda) * refCOV + ...
              regularization_lambda * reg_val * eye(size(refCOV, 1));

[Vref, Dref] = eig(refCOV_reg);
[~, idx]     = sort(diag(Dref), 'descend');
basis_ref    = Vref(:, idx(1:SSI_top_PCs));

%% ── 3. Epoch both data streams ───────────────────────────────────────────
epoch_samples = round(srate * epoch_size);

C_signal   = make_cov_array(signal_data,   epoch_samples);
C_artifact = make_cov_array(artifact_data, epoch_samples);

% Align epoch counts
n_sig  = length(C_signal);
n_art  = length(C_artifact);
n_use  = min(n_sig, n_art);
if n_use < 3          % Not enough epochs for 5-fold CV → bail out
    lda_accuracy = NaN;
    return;
end
C_signal   = C_signal(1:n_use);
C_artifact = C_artifact(1:n_use);

%% ── 4. Extract SSI and log-power per epoch ───────────────────────────────
ssi_signal   = compute_ssi(C_signal,   basis_ref, SSI_top_PCs);
ssi_artifact = compute_ssi(C_artifact, basis_ref, SSI_top_PCs);

lpow_signal   = 10 * log10(cellfun(@trace, C_signal));
lpow_artifact = 10 * log10(cellfun(@trace, C_artifact));

%% ── 5. 2D LDA (5-fold cross-validated) ──────────────────────────────────
X_lda = [ssi_signal(:),   lpow_signal(:); ...
          ssi_artifact(:), lpow_artifact(:)];
Y_lda = [ones(n_use, 1); zeros(n_use, 1)];

try
    lda_mdl      = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5);
    lda_accuracy = (1 - kfoldLoss(lda_mdl)) * 100;
catch
    lda_accuracy = NaN;
end

end % ── end main function ──────────────────────────────────────────────────


%% ══ Local helpers ═════════════════════════════════════════════════════════

function COV_array = make_cov_array(data, epoch_samples)
% Pad data to a multiple of epoch_samples and return cell array of epoch covs.
    pnts = size(data, 2);
    remainder = rem(pnts, epoch_samples);
    if remainder ~= 0
        pad_len = epoch_samples - remainder;
        seg     = data(:, end - pad_len + 1 : end);
        data    = [data, fliplr(seg)];
    end
    n_epochs  = size(data, 2) / epoch_samples;
    COV_array = cell(n_epochs, 1);
    for e = 1:n_epochs
        i1 = (e - 1) * epoch_samples + 1;
        i2 = i1 + epoch_samples - 1;
        COV_array{e} = cov(data(:, i1:i2)');
    end
end


function ssi = compute_ssi(COV_array, basis_ref, top_PCs)
% Compute geometric-mean cosine SSI for each epoch.
    n   = length(COV_array);
    ssi = zeros(n, 1);
    for i = 1:n
        [V, D]   = eig(COV_array{i});
        [~, idx] = sort(diag(D), 'descend');
        basis_e  = V(:, idx(1:top_PCs));
        cos_ang  = subspace_angles(basis_e, basis_ref);   % top_PCs cosines
        ssi(i)   = prod(cos_ang) .^ (1 / top_PCs);
    end
end
