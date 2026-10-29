% Secondary BEAR Benchmark: Real-World Anatomical Perturbations & 3D Cap Shifts.
%
% Simulates realistic mismatches between the true forward physics of the
% ground-truth clean EEG and the standard template leadfield assumed by GEDAI:
%   Regime 0 (Nominal): Unperturbed empirical BEAR clean EEG + artifacts (-9 dB)
%   Regime 1 (Cap Shift 12 deg + 8 mm jitter): 3D rigid cap rotation (pitch/roll/yaw)
%            + local electrode displacement on the 343-channel 10-05 BEM mesh
%   Regime 2 (Cortical Folding + Skull Tilt): Subject-specific dipole/gyral subspace
%            rotation + skull-conductivity harmonic spectral tilt (beta = 0.80)
%   Regime 3 (Severe Combined Shift + Anatomy): 15 deg 3D cap rotation + 10 mm
%            electrode jitter + cortical folding perturbation
%
% All blind and static methods receive ONLY the nominal unshifted chanlocs and
% standard template leadfield (zero knowledge of the perturbation). Also evaluates
% the Oracle Perturbed Prior to measure the fraction of the anatomical gap recovered
% blindly from the contaminated (-9 dB) mixture.

clear;
clc;
maxNumCompThreads(1);

bench_file_dir = fileparts(mfilename('fullpath'));
plugin_dir = fileparts(bench_file_dir);
eeglab_dir = fileparts(fileparts(plugin_dir));
if ~exist(fullfile(eeglab_dir, 'functions'), 'dir')
    eeglab_dir = 'C:\Users\Ros\Documents\MATLAB\eeglab2025.0.0';
    plugin_dir = fullfile(eeglab_dir, 'plugins', 'GEDAI-master');
end

if exist('E:\new 4GEDAI paper\EMPIRICAL analysis\EHRLICH 27 channel', 'dir')
    data_dir = 'E:\new 4GEDAI paper\EMPIRICAL analysis\EHRLICH 27 channel';
    analysis_dir = fullfile(plugin_dir, 'benchmarks');
else
    analysis_dir = 'C:\Users\Ros\Documents\EEG data\BEAR\analysis code';
    data_dir = 'C:\Users\Ros\Documents\EEG data\BEAR\EMPIRICAL analysis';
end
clean_dir = fullfile(data_dir, 'CLEAN EEG');
artifact_dir = fullfile(data_dir, 'ARTIFACTS');

addpath(genpath(fullfile(eeglab_dir, 'functions')));
addpath(plugin_dir);
addpath(fullfile(plugin_dir, 'auxiliaries'));

clean_files = dir(fullfile(clean_dir, '*.set'));
[~, clean_order] = sort({clean_files.name});
clean_files = clean_files(clean_order);
categories = {'EMG', 'EOG', 'NOISE', 'NOISE EOG EMG'};
num_clean = numel(clean_files);
assert(num_clean == 10, 'Expected 10 clean BEAR recordings.');

snr_db = -9;
artifact_scale = 1 / 10^(snr_db / 20);
train_seconds = 30;

clean_data = cell(1, num_clean);
for i = 1:num_clean
    EEG_c = pop_loadset('filename', clean_files(i).name, 'filepath', clean_dir, 'verbose', 'off');
    clean_data{i} = double(single(reshape(zscore(double(EEG_c.data(:))), size(EEG_c.data))));
    if i == 1
        chanlocs = EEG_c.chanlocs;
        srate = EEG_c.srate;
        labels = lower(string({chanlocs.labels}));
    end
end

num_categories = numel(categories);
num_artifacts = num_categories * num_clean;
art_data = cell(1, num_artifacts);
art_category = strings(num_artifacts, 1);
ai = 0;
for c = 1:num_categories
    files = dir(fullfile(artifact_dir, categories{c}, '*.set'));
    [~, order] = sort({files.name});
    files = files(order);
    for i = 1:numel(files)
        ai = ai + 1;
        EEG_art = pop_loadset('filename', files(i).name, 'filepath', files(i).folder, 'verbose', 'off');
        art_data{ai} = double(artifact_scale * single(reshape(zscore(double(EEG_art.data(:))), size(EEG_art.data))));
        art_category(ai) = categories{c};
    end
end

% Load 343-channel 10-05 template leadfield and 3D electrode coordinates
leadfield_file = fullfile(plugin_dir, 'auxiliaries', 'fsavLEADFIELD_4_GEDAI.mat');
L_struct = load(leadfield_file, 'leadfield4GEDAI');
LF = L_struct.leadfield4GEDAI;
template_labels = lower(string({LF.electrodes.Name}));
channel_indices = zeros(1, numel(labels));
for i = 1:numel(labels)
    match = find(template_labels == labels(i), 1);
    if isempty(match)
        match = find(contains(template_labels, labels(i)), 1);
    end
    channel_indices(i) = match;
end

G_343_av = LF.Gain - mean(LF.Gain, 1);
pos_343 = zeros(numel(LF.electrodes), 3);
for i = 1:numel(LF.electrodes)
    pos_343(i, :) = LF.electrodes(i).Loc(:)';
end
% Center 343 electrode coordinates on the sphere center for 3D cap rotations
pos_center = mean(pos_343, 1);
pos_343_centered = pos_343 - pos_center;
head_radius = median(vecnorm(pos_343_centered, 2, 2)); % ~0.095 m (95 mm)

C_LF = LF.gram_matrix_avref(channel_indices, channel_indices);
C_LF = (C_LF + C_LF') / 2;
C_LF_n = C_LF / trace(C_LF);

N = numel(channel_indices);
R_n = eye(N) - ones(N) / N;
R_np1 = eye(N) - ones(N) / (N + 1);
C0n = (R_n * C_LF_n * R_n') / trace(R_n * C_LF_n * R_n');

% Build the 40-mixture balanced benchmark combinations (10 Clean x 4 Categories)
pilot_combos = zeros(40, 2);
p_row = 0;
for c = 1:num_categories
    for ci = 1:num_clean
        p_row = p_row + 1;
        ai = (c - 1) * num_clean + ci;
        pilot_combos(p_row, :) = [ci, ai];
    end
end

regimes = {'Regime0_Nominal_Unperturbed', ...
           'Regime1_CapShift_12deg_8mm', ...
           'Regime2_Anatomy_Folding_Skull', ...
           'Regime3_Combined_Severe_15deg'};
methods = {'Raw', 'Leadfield_343ch_Precomp', 'Leadfield_27ch_AvgRef', ...
           'Blind_Subspace_McCann', 'Oracle_Perturbed_Prior'};

% Precompute subject-specific perturbation operators T_pert{r, s} and C_true{r, s}
T_pert = cell(numel(regimes), num_clean);
C_true_prior = cell(numel(regimes), num_clean);
bw_perturbation = zeros(numel(regimes), num_clean);

for r = 1:numel(regimes)
    for s = 1:num_clean
        [T_pert{r, s}, C_true_prior{r, s}, bw_perturbation(r, s)] = ...
            build_anatomical_perturbation(r, s, C0n, G_343_av, pos_343_centered, channel_indices, head_radius);
    end
end

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');
end

num_trials = size(pilot_combos, 1);
num_regimes = numel(regimes);
num_methods = numel(methods);
all_metrics = zeros(num_trials, num_methods, 3, num_regimes); % trial x method x (Corr, RRMSE, SNR) x regime

bench_timer = tic;
fprintf('Running Secondary Anatomical Perturbation BEAR Benchmark (%d regimes x %d mixtures)...\n', ...
    num_regimes, num_trials);

for r = 1:num_regimes
    T_r = T_pert(r, :);
    C_true_r = C_true_prior(r, :);
    metrics_r = zeros(num_trials, num_methods, 3);

    parfor trial = 1:num_trials
        maxNumCompThreads(1);
        ci = pilot_combos(trial, 1);
        ai = pilot_combos(trial, 2);

        % Apply anatomical / cap-shift transport operator to ground-truth clean EEG
        clean_orig = clean_data{ci};
        clean_shifted = T_r{ci} * clean_orig;
        % Preserve exact unit z-score variance so input SNR remains strictly -9.0 dB
        clean = reshape(zscore(clean_shifted(:)), size(clean_orig));
        artifact = art_data{ai};

        mixture = clean + artifact;
        split_sample = train_seconds * srate;
        train_data = mixture(:, 1:split_sample);
        clean_test = clean(:, split_sample+1:end);
        mixture_test = mixture(:, split_sample+1:end);

        % Blind Subspace McCann tuning on first 30s (given ONLY standard C_LF and chanlocs!)
        [refCOV_blind] = GEDAI_subspace_mccann_prior(train_data, srate, chanlocs, ...
            C_LF, 'AvgRef', 0.5, 12, 'eeg', false);

        % Oracle true perturbed prior in Band-Selective 3D tensor
        refCOV_oracle = make_oracle_3d(C0n, C_true_r{ci}, R_np1);

        m_raw    = score_rec(clean_test, mixture_test);
        out_343  = run_gedai_custom(chanlocs, srate, mixture_test, C_LF);
        out_27   = run_gedai_precomp(chanlocs, srate, mixture_test);
        out_blnd = run_gedai_3d(chanlocs, srate, mixture_test, refCOV_blind, R_np1);
        out_orac = run_gedai_3d(chanlocs, srate, mixture_test, refCOV_oracle, R_np1);

        t_m = zeros(num_methods, 3);
        t_m(1, :) = m_raw;
        t_m(2, :) = score_rec(clean_test, out_343);
        t_m(3, :) = score_rec(clean_test, out_27);
        t_m(4, :) = score_rec(clean_test, out_blnd);
        t_m(5, :) = score_rec(clean_test, out_orac);
        metrics_r(trial, :, :) = t_m;
    end

    all_metrics(:, :, :, r) = metrics_r;
    print_regime_summary(regimes{r}, mean(bw_perturbation(r, :)), metrics_r, ...
        pilot_combos(:, 1), art_category(pilot_combos(:, 2)), methods);
end

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_mat = fullfile(analysis_dir, ['anatomical_perturbation_bear_' timestamp '.mat']);
save(out_mat, 'all_metrics', 'regimes', 'methods', 'bw_perturbation', 'pilot_combos', '-v7.3');
fprintf('\nSaved secondary anatomical perturbation benchmark results to:\n  %s\nElapsed: %.1f min\n', ...
    out_mat, toc(bench_timer) / 60);

function [T_op, C_pert, bw_dist] = build_anatomical_perturbation(regime_idx, subj_idx, C0n, G_343_av, pos_343, chan_idx, head_radius)
N = size(C0n, 1);
R_n = eye(N) - ones(N) / N;
[V_eig, D_eig] = eig(C0n);
[lambda0, ord] = sort(max(real(diag(D_eig)), 0), 'descend');
V0 = real(V_eig(:, ord));
K = 6;
V_K = V0(:, 1:K);
Lambda_K = diag(lambda0(1:K));

if regime_idx == 1
    % Nominal unperturbed
    T_op = eye(N);
    C_pert = C0n;
    bw_dist = 0;
    return;
end

rs = RandStream('mt19937ar', 'Seed', 100 * regime_idx + subj_idx);

if regime_idx == 2 || regime_idx == 4
    % 3D Cap Shift + Local Electrode Jitter on the 343-channel 10-05 mesh
    if regime_idx == 2
        ang_deg = 12.0;
        jit_mm  = 8.0;
    else
        ang_deg = 15.0;
        jit_mm  = 10.0;
    end
    angles = (ang_deg * pi / 180) * (2 * rand(rs, 1, 3) - 1);
    Rx = [1 0 0; 0 cos(angles(1)) -sin(angles(1)); 0 sin(angles(1)) cos(angles(1))];
    Ry = [cos(angles(2)) 0 sin(angles(2)); 0 1 0; -sin(angles(2)) 0 cos(angles(2))];
    Rz = [cos(angles(3)) -sin(angles(3)) 0; sin(angles(3)) cos(angles(3)) 0; 0 0 1];
    R_rot = Rz * Ry * Rx;

    pos_nom = pos_343(chan_idx, :);
    jitter = (jit_mm / 1000) * (head_radius / 0.095) * randn(rs, N, 3);
    pos_shifted = (pos_nom * R_rot') + jitter;

    % Interpolate shifted electrode positions onto the 343-channel 10-05 leadfield via Gaussian kernel
    sigma_d = 0.18 * head_radius;
    W_343 = zeros(N, size(pos_343, 1));
    for i = 1:N
        dists = vecnorm(pos_343 - pos_shifted(i, :), 2, 2);
        w_i = exp(-(dists.^2) / (2 * sigma_d^2));
        W_343(i, :) = w_i / sum(w_i);
    end
    G_shift = R_n * (W_343 * G_343_av);
    C_cap = G_shift * G_shift';
    C_cap = real((C_cap + C_cap') / 2);
    C_cap = C_cap / trace(C_cap);
else
    C_cap = C0n;
end

if regime_idx == 3 || regime_idx == 4
    % Cortical folding / dipole orientation rotation in K=6 harmonic subspace + skull tilt
    L_anat = 0.28;
    th = L_anat * (2 * rand(rs, 1, 12) - 1);
    H = diag([th(1:3), -th(1:3)]);
    H(1,4)=th(4); H(4,1)=th(4); H(2,5)=th(5); H(5,2)=th(5); H(3,6)=th(6); H(6,3)=th(6);
    H(1,5)=th(7); H(5,1)=th(7); H(2,4)=th(8); H(4,2)=th(8); H(1,6)=th(9); H(6,1)=th(9);
    H(1,2)=th(10); H(2,1)=th(10); H(1,3)=th(11); H(3,1)=th(11); H(2,3)=th(12); H(3,2)=th(12);
    M_K = expm(H);
    top_cov = M_K * Lambda_K * M_K';
    M_K = sqrt(trace(Lambda_K) / trace(top_cov)) * M_K;
    T_fold = eye(N) + V_K * (M_K - eye(K)) * V_K';
    C_pert = T_fold * C_cap * T_fold';
    C_pert = real((C_pert + C_pert') / 2);
    C_pert = C_pert / trace(C_pert);
else
    C_pert = C_cap;
end

% Construct transport operator T_op in the K=6 subspace mapping C0n -> C_pert
G_0 = V_K' * C0n * V_K;
G_p = V_K' * C_pert * V_K;
G_p = (G_p + G_p') / 2;
M_transport = sqrtm(G_p) * inv(sqrtm(G_0));
M_transport = real(M_transport);
T_op = R_n * (eye(N) + V_K * (M_transport - eye(K)) * V_K') * R_n;

% Compute Bures-Wasserstein distance between C0n and C_pert
root0 = V0 * diag(sqrt(max(lambda0, 0))) * V0';
mid = real((root0 * C_pert * root0 + (root0 * C_pert * root0)') / 2);
bw_dist = sqrt(max(0, 2 - 2 * sum(sqrt(max(real(eig(mid)), 0)))));
end

function refCOV_3D = make_oracle_3d(C0n, C_true, R_np1)
refCOV_3D = repmat(C0n, 1, 1, 8);
C_mid = 0.50 * C_true + 0.50 * C0n;
C_mid = C_mid / trace(C_mid);
for b = 3:6
    refCOV_3D(:, :, b) = C_mid;
end
C_7 = 0.18 * C_true + 0.82 * C0n;
C_7 = C_7 / trace(C_7);
refCOV_3D(:, :, 7) = C_7;
for b = 1:8
    refCOV_3D(:, :, b) = R_np1 * refCOV_3D(:, :, b) * R_np1';
    refCOV_3D(:, :, b) = real((refCOV_3D(:, :, b) + refCOV_3D(:, :, b)') / 2);
end
end

function metric = score_rec(clean, estimate)
clean = double(clean);
estimate = double(estimate);
residual = estimate - clean;
c = corrcoef(clean(:), estimate(:));
rrmse = sqrt(mean(residual(:).^2)) / max(sqrt(mean(clean(:).^2)), 1e-12);
snr = 10 * log10(var(clean(:)) / max(var(residual(:)), 1e-12));
metric = [c(1, 2), rrmse, snr];
end

function out = run_gedai_custom(chanlocs, srate, data, refCOV)
EEG = struct('setname','p','data',data,'srate',srate,'chanlocs',chanlocs,...
    'nbchan',size(data,1),'pnts',size(data,2),'trials',1,'xmin',0,'xmax',(size(data,2)-1)/srate,'etc',struct());
EEG_c = GEDAI(EEG, 'auto', 12, 0.5, refCOV, false, false, inf, inf, 'eeg', Inf, 'AvgRef', struct('silent', true));
out = double(EEG_c.data);
end

function out = run_gedai_precomp(chanlocs, srate, data)
EEG = struct('setname','p','data',data,'srate',srate,'chanlocs',chanlocs,...
    'nbchan',size(data,1),'pnts',size(data,2),'trials',1,'xmin',0,'xmax',(size(data,2)-1)/srate,'etc',struct());
EEG_c = GEDAI(EEG, 'auto', 12, 0.5, 'precomputed', false, false, inf, inf, 'eeg', Inf, 'AvgRef', struct('silent', true));
out = double(EEG_c.data);
end

function out = run_gedai_3d(chanlocs, srate, data, refCOV_3D, R_np1)
EEG = struct('setname','p','data',R_np1*data,'srate',srate,'chanlocs',chanlocs,...
    'nbchan',size(data,1),'pnts',size(data,2),'trials',1,'xmin',0,'xmax',(size(data,2)-1)/srate,'etc',struct(),'ref','average');
EEG_c = GEDAI(EEG, 'auto', 12, 0.5, refCOV_3D, false, false, inf, inf, 'eeg', Inf, 'None', struct('silent', true));
out = double(EEG_c.data);
end

function print_regime_summary(regime_name, mean_bw, m_sub, clean_idx, ~, methods)
n_t = size(m_sub, 1);
fprintf('\n====================================================================\n');
fprintf('%s (Mean Perturbation BW Dist = %.4f, N = %d)\n', regime_name, mean_bw, n_t);
fprintf('====================================================================\n');
for m = 1:numel(methods)
    fprintf('%-26s Corr %.4f | RRMSE %.4f | SNR %.3f dB\n', methods{m}, ...
        mean(m_sub(:, m, 1)), mean(m_sub(:, m, 2)), mean(m_sub(:, m, 3)));
end
d_blind_vs_27 = m_sub(:, 4, 3) - m_sub(:, 3, 3);
d_orac_vs_27  = m_sub(:, 5, 3) - m_sub(:, 3, 3);
[~, p_t, ci] = ttest(d_blind_vs_27);
p_sr = signrank(d_blind_vs_27);
subj_d = arrayfun(@(s) mean(d_blind_vs_27(clean_idx == s)), 1:10);
fprintf('  Blind vs Static C0n: mean=%+.3f dB, 95%% CI=[%+.3f,%+.3f], wins=%d/%d (Subj wins=%d/10), t-p=%.4g, sr-p=%.4g\n', ...
    mean(d_blind_vs_27), ci(1), ci(2), sum(d_blind_vs_27 > 0), n_t, sum(subj_d > 0), p_t, p_sr);
if mean(d_orac_vs_27) > 0.01
    fprintf('  Oracle Recovery Ratio: %.1f%% of Oracle Perturbed Prior gain (%+.3f dB / %+.3f dB)\n', ...
        100 * mean(d_blind_vs_27) / mean(d_orac_vs_27), mean(d_blind_vs_27), mean(d_orac_vs_27));
end
end
