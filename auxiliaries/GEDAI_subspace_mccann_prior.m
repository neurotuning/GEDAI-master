function [refCOV_3D, best_params, best_SENSAI, base_SENSAI, bw_distance] = GEDAI_subspace_mccann_prior(...
    data, srate, ~, C_LF, reference_mode, lowcut_frequency, ~, signal_type, verbose)
% GEDAI_SUBSPACE_MCCANN_PRIOR - Generative Semi-Blind Subspace McCann
% leadfield tuning for GEDAI (maximizing mid-band SENSAI without empirical
% covariance estimation).
%
% Perturbs the theoretical leadfield prior strictly within the span of the
% top K = 6 smooth cortical harmonic modes V_K via a low-rank McCann
% displacement operator:
%   T(theta) = I + V_K * (M_K(theta) - I_K) * V_K'
%   C_trial(theta) = T(theta) * C0n * T(theta)'
% where M_K(theta) = expm(H_K(theta)) is parameterized by 9 tangent
% generators (3 spectral tilts + 6 off-diagonal couplings between modes
% 1..3 and 4..6), while the bottom N - K high-spatial-frequency modes are
% strictly frozen.
%
% Evaluates candidate leadfield perturbations purely by maximizing the
% SENSAI score on 1-second 8-16 Hz calibration epochs (Guardrail A), and
% constructs a Band-Selective 3D reference covariance tensor:
%   - Outer wavelet bands (> 25 Hz EMG/gamma and < 1.56 Hz EOG/delta)
%     retain the unperturbed anatomical leadfield prior C_LF_n.
%   - Physiological mid-bands (1.56 - 25 Hz) use the anchor-regularized
%     subspace McCann covariance (w * C_K(theta*) + (1 - w) * C_LF_n).
%
% Inputs:
%   data                 - (nbchan x pnts) referenced EEG data (e.g. EEGavRef.data)
%   srate                - Sampling rate in Hz
%   chanlocs             - EEG.chanlocs struct (unused, kept for signature compatibility)
%   C_LF                 - Theoretical leadfield covariance prior (nbchan x nbchan)
%   reference_mode       - Reference mode ('AvgRef', 'REST', or channel label)
%   lowcut_frequency     - Low-cut frequency in Hz (default = 0.5)
%   epoch_size_in_cycles - Wavelet epoch size in cycles (default = 12)
%   signal_type          - 'eeg' or 'meg' (default = 'eeg')
%   verbose              - Boolean/numeric to print calibration summary (default = true)
%
% Outputs:
%   refCOV_3D   - (nbchan x nbchan x num_bands) band-selective reference covariance tensor
%   best_params - Selected 1x9 tangent parameter vector theta*
%   best_SENSAI - Maximal 8-16 Hz SENSAI score (%)
%   base_SENSAI - Baseline leadfield 8-16 Hz SENSAI score (%)
%   bw_distance - Bures-Wasserstein distance between C0n and C_K(theta*)

if nargin < 5 || isempty(reference_mode), reference_mode = 'AvgRef'; end
if nargin < 6 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 8 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 9 || isempty(verbose), verbose = true; end

data = double(data);
[N, pnts] = size(data);

% Subsample up to 180 seconds for fast calibration on long recordings
max_cal_pnts = round(180 * srate);
if pnts <= max_cal_pnts
    cal_data = data;
else
    seg_len = round(30 * srate);
    idx1 = 1:seg_len;
    mid_start = round(pnts / 2 - seg_len / 2);
    idx2 = mid_start:(mid_start + seg_len - 1);
    idx3 = (pnts - seg_len + 1):pnts;
    cal_data = [data(:, idx1), data(:, idx2), data(:, idx3)];
end

% Unit-trace theoretical leadfield prior (preserving original trace scale)
C_LF = real((C_LF + C_LF') / 2);
orig_trace = max(trace(C_LF), 1e-12);
C_LF_n = C_LF / orig_trace;

% Construct reference-matched anchor C0n and harmonic eigenbasis V0
if strcmpi(reference_mode, 'AvgRef')
    R_proj = eye(N) - ones(N) / N;
    cal_data_proj = R_proj * cal_data;
    C0n = (R_proj * C_LF_n * R_proj') / max(trace(R_proj * C_LF_n * R_proj'), 1e-12);
else
    cal_data_proj = cal_data;
    C0n = C_LF_n;
end

[V_eig, D_eig] = eig(C0n);
[lambda0, eig_order] = sort(max(real(diag(D_eig)), 0), 'descend');
V0 = real(V_eig(:, eig_order));

K = min(6, max(2, floor(N / 2)));
V_K = V0(:, 1:K);
Lambda_K = diag(lambda0(1:K));

% 1. Extract 8-16 Hz physiological mid-band calibration epochs (Guardrail A)
[fb, fa] = butter(4, [8, min(16, 0.45 * srate)] / (srate / 2), 'bandpass');
alpha_cal = filtfilt(fb, fa, cal_data_proj')';

epoch_len = round(srate);
n_epochs = floor(size(alpha_cal, 2) / epoch_len);
best_params = zeros(1, 9);

if n_epochs >= 2 && K >= 6
    epoched = reshape(double(alpha_cal(:, 1:(n_epochs * epoch_len))), N, epoch_len, n_epochs);
    COV_tr = zeros(N, N, n_epochs);
    for e = 1:n_epochs
        cv = cov(epoched(:, :, e)');
        COV_tr(:, :, e) = (cv + cv') / 2;
    end

    % 2. Deterministic Trust-Region Design in K=6 Tangent Space (L_max = 0.22)
    L_max = 0.22;
    rs = RandStream('mt19937ar', 'Seed', 1);
    P_cand = [zeros(1, 9); ...
              0.6 * L_max * eye(9); ...
             -0.6 * L_max * eye(9); ...
              L_max * (2 * rand(rs, 60, 9) - 1)];

    num_cands = size(P_cand, 1);
    scores = zeros(num_cands, 1);
    C_cands = cell(num_cands, 1);

    for p_idx = 1:num_cands
        C_c = make_subspace_candidate(C0n, V_K, Lambda_K, P_cand(p_idx, :));
        C_cands{p_idx} = C_c;
        scores(p_idx) = eval_midband_sensai(C_c, COV_tr, signal_type);
    end

    base_SENSAI = scores(1);
    [best_SENSAI, best_idx] = max(scores);
    if best_SENSAI > base_SENSAI
        C_K = C_cands{best_idx};
        best_params = P_cand(best_idx, :);
    else
        best_SENSAI = base_SENSAI;
        C_K = C0n;
    end
else
    base_SENSAI = NaN;
    best_SENSAI = NaN;
    C_K = C0n;
end

bw_distance = compute_bures_distance(C0n, C_K);

% 3. Construct Band-Selective 3D refCOV Tensor
% Outer bands (> 25 Hz and < 1.56 Hz) retain C_LF_n;
% Physiological mid-bands (1.56 - 25 Hz) use anchor-regularized C_K
num_wavelet_bands = max(6, min(ceil(log2(srate / lowcut_frequency)), floor(log2(pnts))));
lower_freqs = srate ./ (2.^(2:(num_wavelet_bands + 1)));
upper_freqs = srate ./ (2.^(1:num_wavelet_bands));
center_freqs = (lower_freqs + upper_freqs) / 2;

lowest_to_exclude = sum(upper_freqs <= lowcut_frequency);
n_total_bands = max(8, num_wavelet_bands - lowest_to_exclude);
refCOV_3D = repmat(C_LF_n, 1, 1, n_total_bands);
adapt_bands = find(center_freqs(1:n_total_bands) >= 1.5 & center_freqs(1:n_total_bands) <= 25.0);
if isempty(adapt_bands)
    adapt_bands = intersect(3:6, 1:n_total_bands);
end

w_subspace = 0.66;
C_mid = w_subspace * C_K + (1 - w_subspace) * C_LF_n;

for idx_b = 1:numel(adapt_bands)
    f = adapt_bands(idx_b);
    refCOV_3D(:, :, f) = C_mid;
end

% Restore original leadfield trace scale so log(Eval) + 100 thresholding in
% SENSAI.m / clean_EEG.m is invariant across raw vs z-scored uV units.
refCOV_3D = refCOV_3D * orig_trace;

if verbose
    fprintf('\n=== GEDAI: Generative Semi-Blind McCann Tuning (K = %d/%d modes) ===\n', K, N);
    fprintf('  Baseline 8-16 Hz SENSAI:    %+.2f %%\n', base_SENSAI);
    fprintf('  Tuned 8-16 Hz SENSAI:       %+.2f %% (BW dist = %.4f)\n', best_SENSAI, bw_distance);
    fprintf('  Band-Selective 3D Prior:    Adapted bands [%s], Frozen outer bands\n', num2str(adapt_bands));
    fprintf('===================================================================\n');
end

end

function C_cand = make_subspace_candidate(C0n, V_K, Lambda_K, theta)
N = size(C0n, 1);
K = size(V_K, 2);
H = diag([theta(1:3), -theta(1:3)]);
H(1, 4) = theta(4); H(4, 1) = theta(4);
H(2, 5) = theta(5); H(5, 2) = theta(5);
H(3, 6) = theta(6); H(6, 3) = theta(6);
H(1, 5) = theta(7); H(5, 1) = theta(7);
H(2, 4) = theta(8); H(4, 2) = theta(8);
H(1, 6) = theta(9); H(6, 1) = theta(9);

M_K = expm(H);
top_cov = M_K * Lambda_K * M_K';
M_K = sqrt(trace(Lambda_K) / max(trace(top_cov), 1e-12)) * M_K;
T = eye(N) + V_K * (M_K - eye(K)) * V_K';
C_cand = T * C0n * T';
C_cand = (C_cand + C_cand') / 2;
C_cand = C_cand / max(trace(C_cand), 1e-12);
end

function score = eval_midband_sensai(refCOV, COV, signal_type)
N = size(refCOV, 1);
reg_lambda = 0.05;
refCOV_reg = (1 - reg_lambda) * refCOV + reg_lambda * (1 / N) * eye(N);
[V, D] = eig(refCOV_reg);
[~, order] = sort(real(diag(D)), 'descend');
top_pcs = min(3, N);
template = V(:, order(1:top_pcs));

n_epochs = size(COV, 3);
Evec = zeros(N, N, n_epochs);
Eval = zeros(N, N, n_epochs);
for e = 1:n_epochs
    [Evec(:, :, e), Eval(:, :, e)] = eig(COV(:, :, e), refCOV_reg, 'chol');
end
[~, score] = SENSAI_fminbnd(-6, 12, refCOV, Eval, Evec, 3, COV, template, signal_type, top_pcs);
end

function dist = compute_bures_distance(C0, C1)
C0 = real((C0 + C0') / 2);
C1 = real((C1 + C1') / 2);
C0 = C0 / max(trace(C0), 1e-12);
C1 = C1 / max(trace(C1), 1e-12);
[V, D] = eig(C0);
root0 = V * diag(sqrt(max(real(diag(D)), 0))) * V';
mid = root0 * C1 * root0;
mid = real((mid + mid') / 2);
fid_root = sum(sqrt(max(real(eig(mid)), 0)));
dist = sqrt(max(0, 2 - 2 * fid_root));
end
