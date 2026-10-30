function [refCOV_3D, best_params, best_SENSAI, base_SENSAI, bw_distance, C_broadband] = GEDAI_subspace_mccann_prior(...
    data, srate, ~, C_LF, reference_mode, lowcut_frequency, ~, signal_type, verbose)
% GEDAI_SUBSPACE_MCCANN_PRIOR - Closed-Form Exact Subspace McCann Geodesic
% Leadfield Tuning for GEDAI with Smooth-Cortical Alpha Epoch Screening and
% Band-Selective 3D Prior.
%
% Transports the theoretical leadfield prior strictly within the span of the
% top K = min(9, max(6, floor(N / 3))) smooth cortical harmonic modes V_K
% via the exact closed-form Bures-Wasserstein (McCann) optimal transport map:
%   A = Lambda_K,  B = (trace(Lambda_K) / trace(S_emp_K)) * (V_K' * C_emp * V_K)
%   T_AB = A^(-1/2) * (A^(1/2) * B * A^(1/2))^(1/2) * A^(-1/2)
%   M_K(t) = (1 - t) * I_K + t * T_AB
%   T(t)   = I_N + V_K * (M_K(t) - I_K) * V_K'
%   C_K(t) = T(t) * C0n * T(t)'
% while the bottom N - K high-spatial-frequency modes remain strictly frozen
% at their anatomical eigenvalues, preventing any adaptation to focal muscle,
% ocular, or single-electrode artifacts.
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
%   best_params - Selected geodesic step t_mccann in [0, 1]
%   best_SENSAI - Maximal 8-13 Hz SENSAI score (%) on screened calibration epochs
%   base_SENSAI - Baseline leadfield 8-13 Hz SENSAI score (%) on screened calibration epochs
%   bw_distance - Bures-Wasserstein distance between C0n and C_K(t*)
%   C_broadband - Broadband reference covariance matrix matched to the mid-band prior

if nargin < 5 || isempty(reference_mode), reference_mode = 'AvgRef'; end
if nargin < 6 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 8 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 9 || isempty(verbose), verbose = true; end

data = double(data);
[N, pnts] = size(data);
orig_trace = max(trace(real((C_LF + C_LF') / 2)), 1e-12);

% Unit-trace theoretical leadfield prior (already referenced in GEDAI.m)
C0n = real((C_LF + C_LF') / 2) / orig_trace;

[V_eig, D_eig] = eig(C0n);
[lambda0, eig_order] = sort(max(real(diag(D_eig)), 0), 'descend');
V0 = real(V_eig(:, eig_order));

K = min(9, max(6, floor(N / 3)));
K = min(K, max(1, N - 2));
V_K = V0(:, 1:K);
Lambda_K = diag(max(lambda0(1:K), 1e-12));

% 1. Extract 8-13 Hz physiological alpha/mu band across the recording
f_lo = min(8.0, 0.30 * srate);
f_hi = min(13.0, 0.45 * srate);
if f_hi > f_lo + 0.5
    [fb, fa] = butter(4, [f_lo, f_hi] / (srate / 2), 'bandpass');
    alpha_cal = filtfilt(fb, fa, data')';
else
    alpha_cal = data;
end

epoch_len = max(10, round(srate));
n_epochs = floor(size(alpha_cal, 2) / epoch_len);
t_mccann = 0.80;

if n_epochs >= 4 && K >= 3
    epoched_al = reshape(double(alpha_cal(:, 1:(n_epochs * epoch_len))), N, epoch_len, n_epochs);
    epoched_bb = reshape(double(data(:, 1:(n_epochs * epoch_len))), N, epoch_len, n_epochs);
    COV_tr = zeros(N, N, n_epochs);
    alpha_pow = zeros(1, n_epochs);
    bb_pow = zeros(1, n_epochs);
    smooth_frac = zeros(1, n_epochs);

    for e = 1:n_epochs
        ca = cov(epoched_al(:, :, e)');
        ca = (ca + ca') / 2;
        COV_tr(:, :, e) = ca;
        tr_a = max(trace(ca), 1e-12);
        alpha_pow(e) = tr_a;
        smooth_frac(e) = trace(V_K' * ca * V_K) / tr_a;
        bb_pow(e) = trace(cov(epoched_bb(:, :, e)'));
    end

    % Screen for dipolar, high-alpha, non-burst epochs:
    %   - Smoothness in top-K leadfield subspace >= median
    %   - Alpha power >= median
    %   - Exclude top 15% broadband artifact burst epochs
    keep_mask = (smooth_frac >= prctile(smooth_frac, 50)) & ...
                (alpha_pow   >= prctile(alpha_pow, 45)) & ...
                (bb_pow      <= prctile(bb_pow, 85));
    if sum(keep_mask) < 4
        keep_mask = (smooth_frac >= prctile(smooth_frac, 40)) & (bb_pow <= prctile(bb_pow, 85));
    end
    if sum(keep_mask) < 2
        keep_mask = true(1, n_epochs);
    end

    COV_cal = COV_tr(:, :, keep_mask);
    C_alpha_emp = mean(COV_cal, 3);
    C_alpha_emp = (C_alpha_emp + C_alpha_emp') / 2;
    C_alpha_emp = C_alpha_emp / max(trace(C_alpha_emp), 1e-12);

    % 2. Closed-Form Subspace McCann Transport Map on SPD(K)
    S_emp_K = V_K' * C_alpha_emp * V_K;
    S_emp_K = (S_emp_K + S_emp_K') / 2;
    S_emp_K_scaled = S_emp_K * (trace(Lambda_K) / max(trace(S_emp_K), 1e-12));

    A_half = diag(sqrt(diag(Lambda_K)));
    A_inv_half = diag(1 ./ sqrt(diag(Lambda_K)));
    mid_mat = A_half * S_emp_K_scaled * A_half;
    mid_mat = real((mid_mat + mid_mat') / 2);
    [Vm, Dm] = eig(mid_mat);
    mid_sqrt = real(Vm) * diag(sqrt(max(real(diag(Dm)), 0))) * real(Vm)';
    T_AB = A_inv_half * mid_sqrt * A_inv_half;

    % Evaluate baseline and geodesic step t_mccann = 0.80
    base_SENSAI = eval_midband_sensai(C0n, COV_cal, signal_type);
    M_K = (1 - t_mccann) * eye(K) + t_mccann * T_AB;
    T_op = eye(N) + V_K * (M_K - eye(K)) * V_K';
    C_K = T_op * C0n * T_op';
    C_K = real((C_K + C_K') / 2);
    C_K = C_K / max(trace(C_K), 1e-12);
    best_SENSAI = eval_midband_sensai(C_K, COV_cal, signal_type);
    best_params = t_mccann;
else
    base_SENSAI = NaN;
    best_SENSAI = NaN;
    C_K = C0n;
    t_mccann = 0;
    best_params = 0;
end

bw_distance = compute_bures_distance(C0n, C_K);

% 3. Construct Band-Selective 3D refCOV Tensor & Matched Broadband Prior
num_wavelet_bands = max(6, min(ceil(log2(srate / lowcut_frequency)), floor(log2(pnts))));
lower_freqs = srate ./ (2.^(2:(num_wavelet_bands + 1)));
upper_freqs = srate ./ (2.^(1:num_wavelet_bands));

lowest_to_exclude = sum(upper_freqs <= lowcut_frequency);
n_total_bands = max(6, num_wavelet_bands - lowest_to_exclude);

C0_scaled = C0n * orig_trace;
CK_scaled = C_K * orig_trace;

refCOV_3D = repmat(C0_scaled, 1, 1, n_total_bands);
for f = 1:n_total_bands
    fl = lower_freqs(f);
    fu = upper_freqs(f);
    % Full McCann protection in alpha/theta-alpha overlap bands (e.g. 6-13 Hz)
    if fl < 13.0 && fu > 6.0
        w_f = 1.00;
    % Graded protection in adjacent beta/theta bands (3-25 Hz)
    elseif fl < 25.0 && fu > 3.0
        w_f = 0.60;
    % Keep high-frequency (>25 Hz) and sub-delta (<3 Hz) anchored to C_LF
    else
        w_f = 0.00;
    end
    if w_f > 0
        C_f = (1 - w_f) * C0_scaled + w_f * CK_scaled;
        C_f = real((C_f + C_f') / 2);
        refCOV_3D(:, :, f) = C_f * (orig_trace / max(trace(C_f), 1e-12));
    end
end

C_broadband = CK_scaled;

if verbose
    fprintf('\n=== GEDAI: Closed-Form Subspace McCann Geodesic (K = %d/%d modes, t = %.2f) ===\n', K, N, t_mccann);
    fprintf('  Baseline 8-13 Hz SENSAI:    %+.2f %%\n', base_SENSAI);
    fprintf('  Tuned 8-13 Hz SENSAI:       %+.2f %% (BW dist = %.4f)\n', best_SENSAI, bw_distance);
    fprintf('=============================================================================\n');
end

end

function score = eval_midband_sensai(refCOV, COV, signal_type)
N = size(refCOV, 1);
reg_lambda = 0.05;
refCOV_reg = (1 - reg_lambda) * refCOV + reg_lambda * (trace(refCOV) / N) * eye(N);
refCOV_reg = real((refCOV_reg + refCOV_reg') / 2);
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
[~, score] = SENSAI_fminbnd(0, 12, refCOV, Eval, Evec, 3, COV, template, signal_type, top_pcs);
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
