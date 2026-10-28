% BEAR empirical-data benchmark for Generative Semi-Blind Subspace McCann Leadfield Tuning.
% Purely generative K=6 leadfield harmonic perturbation maximizing 8-16 Hz SENSAI
% (without constructing any empirical signal covariance C_S or C_EC), combined
% with Band-Selective 3D refCOV.
% Calibrates on the first 30 seconds and evaluates reconstruction accuracy on
% the untouched final 30 seconds across all 400 clean x artifact mixtures
% (and the 40-mixture balanced pilot).

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
assert(exist('pop_loadset', 'file') == 2, 'EEGLAB pop_loadset was not found.');
assert(exist('GEDAI', 'file') == 2, 'GEDAI was not found.');
assert(exist('GEDAI_subspace_mccann_prior', 'file') == 2, 'GEDAI_subspace_mccann_prior was not found.');

clean_files = dir(fullfile(clean_dir, '*.set'));
[~, clean_order] = sort({clean_files.name});
clean_files = clean_files(clean_order);
categories = {'EMG', 'EOG', 'NOISE', 'NOISE EOG EMG'};
assert(numel(clean_files) == 10, 'Expected 10 clean BEAR recordings.');

snr_db = -9;
artifact_scale = 1 / 10^(snr_db / 20);
train_seconds = 30;
subspace_K = 6;
methods = {'Raw', 'Leadfield_27ch_AvgRef', 'Leadfield_343ch_Precomp', 'Generative_Subspace_McCann'};

% Load the 10 clean EEG recordings and verify shared channel order and sampling rate.
num_clean = numel(clean_files);
clean_data = cell(1, num_clean);
clean_eeg = cell(1, num_clean);
for i = 1:num_clean
    clean_eeg{i} = pop_loadset('filename', clean_files(i).name, ...
        'filepath', clean_dir, 'verbose', 'off');
    clean_data{i} = single(reshape(zscore(double(clean_eeg{i}.data(:))), ...
        size(clean_eeg{i}.data)));
end
chanlocs = clean_eeg{1}.chanlocs;
srate = clean_eeg{1}.srate;
labels = lower(string({chanlocs.labels}));
for i = 1:num_clean
    assert(clean_eeg{i}.srate == srate, 'Clean recordings have different sampling rates.');
    assert(isequal(lower(string({clean_eeg{i}.chanlocs.labels})), labels), ...
        'Clean recordings have different channel orders.');
end

% Load the 40 artifact recordings (10 per artifact category).
num_categories = numel(categories);
num_artifacts = num_categories * num_clean;
art_data = cell(1, num_artifacts);
art_category = strings(num_artifacts, 1);
art_index_in_cat = zeros(num_artifacts, 1);
ai = 0;
for c = 1:num_categories
    files = dir(fullfile(artifact_dir, categories{c}, '*.set'));
    assert(numel(files) == num_clean, ...
        'Expected %d %s artifact files.', num_clean, categories{c});
    [~, order] = sort({files.name});
    files = files(order);
    for i = 1:numel(files)
        ai = ai + 1;
        EEG_art = pop_loadset('filename', files(i).name, ...
            'filepath', files(i).folder, 'verbose', 'off');
        assert(EEG_art.srate == srate, 'Artifact and clean sampling rates differ.');
        assert(isequal(lower(string({EEG_art.chanlocs.labels})), labels), ...
            'Artifact and clean channel orders differ.');
        art_data{ai} = artifact_scale * single(reshape(zscore(double(EEG_art.data(:))), ...
            size(EEG_art.data)));
        art_category(ai) = categories{c};
        art_index_in_cat(ai) = i;
    end
end

% Match GEDAI's precomputed leadfield prior on the 27-channel montage.
leadfield_file = fullfile(plugin_dir, 'auxiliaries', 'fsavLEADFIELD_4_GEDAI.mat');
leadfield = load(leadfield_file, 'leadfield4GEDAI');
template_labels = lower(string({leadfield.leadfield4GEDAI.electrodes.Name}));
channel_indices = zeros(1, numel(labels));
for i = 1:numel(labels)
    match = find(template_labels == labels(i), 1);
    if isempty(match)
        match = find(contains(template_labels, labels(i)), 1);
    end
    assert(~isempty(match), 'No precomputed leadfield channel matches %s.', labels(i));
    channel_indices(i) = match;
end
C_LF = leadfield.leadfield4GEDAI.gram_matrix_avref(channel_indices, channel_indices);
C_LF = (C_LF + C_LF') / 2;
C_LF_n = C_LF / trace(C_LF);

N = size(C_LF, 1);
R_n = eye(N) - ones(N) / N;
R_np1 = eye(N) - ones(N) / (N + 1);
C0n = (R_n * C_LF_n * R_n') / trace(R_n * C_LF_n * R_n');

% Enumerate all 400 (Clean, Artifact) mixtures and mark the 40-mixture balanced pilot.
combos = zeros(num_clean * num_artifacts, 2);
row = 0;
for ci = 1:num_clean
    for ai = 1:num_artifacts
        row = row + 1;
        combos(row, :) = [ci, ai];
    end
end
num_trials = size(combos, 1);
combo_clean = combos(:, 1);
combo_art = combos(:, 2);
pilot_indices = zeros(num_clean * num_categories, 1);
p_row = 0;
for c = 1:num_categories
    for ci = 1:num_clean
        p_row = p_row + 1;
        ai = (c - 1) * num_clean + ci;
        pilot_indices(p_row) = find(combo_clean == ci & combo_art == ai, 1);
    end
end

num_methods = numel(methods);
metrics = nan(num_trials, num_methods, 3); % correlation, relative RMSE, output SNR (dB)
train_sensai = nan(num_trials, 2);
selected_parameters = zeros(num_trials, 9);
selected_distance = nan(num_trials, 1);
trial_clean = combo_clean;
trial_artifact = art_index_in_cat(combo_art);
trial_category = art_category(combo_art);
is_pilot = ismember((1:num_trials)', pilot_indices);

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');
end
C_clean_const = parallel.pool.Constant(clean_data);
C_art_const = parallel.pool.Constant(art_data);

bench_timer = tic;
fprintf('BEAR Generative Subspace McCann Benchmark: %d total mixtures (%d pilot), %.0f dB input SNR, K=%d modes.\n', ...
    num_trials, numel(pilot_indices), snr_db, subspace_K);

parfor trial = 1:num_trials
    maxNumCompThreads(1);
    ci = combo_clean(trial);
    ai = combo_art(trial);
    clean = double(C_clean_const.Value{ci});
    artifact = double(C_art_const.Value{ai});

    mixture = clean + artifact;
    split_sample = train_seconds * srate;
    train_data = mixture(:, 1:split_sample);
    clean_test = clean(:, split_sample+1:end);
    mixture_test = mixture(:, split_sample+1:end);

    % 1. Purely Generative K=6 Subspace McCann Tuning on the first 30 seconds
    % (maximizing 8-16 Hz SENSAI without any empirical C_S or C_EC covariance).
    [refCOV_3D, theta_star, tuned_sc, base_sc, bw_dist] = ...
        GEDAI_subspace_mccann_prior(train_data, srate, chanlocs, ...
        C_LF, 'AvgRef', 0.5, 12, 'eeg', false);
    train_sensai(trial, :) = [base_sc, tuned_sc];
    selected_parameters(trial, :) = theta_star;
    selected_distance(trial) = bw_dist;

    % 2. Evaluate reconstructions on the untouched final 30 seconds.
    raw_metric = score_reconstruction(clean_test, mixture_test);
    out_27ch = denoise_with_custom_avgref(chanlocs, srate, mixture_test, C0n);
    out_precomp = denoise_with_precomputed(chanlocs, srate, mixture_test);
    out_tuned = denoise_with_3d_refcov(chanlocs, srate, mixture_test, refCOV_3D, R_np1);

    m_27ch = score_reconstruction(clean_test, out_27ch);
    m_precomp = score_reconstruction(clean_test, out_precomp);
    m_tuned = score_reconstruction(clean_test, out_tuned);

    trial_metrics = zeros(num_methods, 3);
    trial_metrics(1, :) = raw_metric;
    trial_metrics(2, :) = m_27ch;
    trial_metrics(3, :) = m_precomp;
    trial_metrics(4, :) = m_tuned;
    metrics(trial, :, :) = trial_metrics;
end

% Build full 400-mixture and 40-mixture pilot tables.
raw_metrics = squeeze(metrics(:, 1, :));
base27_metrics = squeeze(metrics(:, 2, :));
precomp_metrics = squeeze(metrics(:, 3, :));
tuned_metrics = squeeze(metrics(:, 4, :));

results_full = table(trial_clean, trial_artifact, trial_category, is_pilot, ...
    raw_metrics(:, 1), raw_metrics(:, 2), raw_metrics(:, 3), ...
    base27_metrics(:, 1), base27_metrics(:, 2), base27_metrics(:, 3), ...
    precomp_metrics(:, 1), precomp_metrics(:, 2), precomp_metrics(:, 3), ...
    tuned_metrics(:, 1), tuned_metrics(:, 2), tuned_metrics(:, 3), ...
    train_sensai(:, 1), train_sensai(:, 2), selected_distance, ...
    'VariableNames', {'CleanIndex', 'ArtifactIndex', 'ArtifactType', 'IsPilot', ...
    'Raw_Correlation', 'Raw_RRMSE', 'Raw_SNR_dB', ...
    'Baseline_Correlation', 'Baseline_RRMSE', 'Baseline_SNR_dB', ...
    'Precomp_Correlation', 'Precomp_RRMSE', 'Precomp_SNR_dB', ...
    'Tuned_Correlation', 'Tuned_RRMSE', 'Tuned_SNR_dB', ...
    'Calibration_SENSAI_Baseline', 'Calibration_SENSAI_Tuned', 'Selected_BW_Distance'});

results_pilot = results_full(pilot_indices, :);

print_benchmark_summary('40-MIXTURE BALANCED PILOT (PURELY GENERATIVE K=6)', metrics(pilot_indices, :, :), ...
    trial_clean(pilot_indices), trial_category(pilot_indices), methods, categories);
print_benchmark_summary('400-MIXTURE FULL BENCHMARK (PURELY GENERATIVE K=6)', metrics, ...
    trial_clean, trial_category, methods, categories);

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
pilot_stem = fullfile(analysis_dir, ['semiblind_leadfield_pilot_' timestamp]);
full_stem = fullfile(analysis_dir, ['semiblind_leadfield_full400_' timestamp]);
writetable(results_pilot, [pilot_stem '.csv']);
writetable(results_full, [full_stem '.csv']);

config = struct('snr_db', snr_db, 'train_seconds', train_seconds, ...
    'subspace_K', subspace_K, 'frequency_band_hz', [8, 16], ...
    'physiological_bands', [3, 4, 5, 6], 'method_names', {methods});
results = results_pilot;
save([pilot_stem '.mat'], 'results', 'results_pilot', 'results_full', 'config', 'selected_parameters', '-v7.3');
fprintf('\nSaved pilot & full results:\n  %s.csv\n  %s.csv\n  %s.mat\nElapsed: %.1f min\n', ...
    pilot_stem, full_stem, pilot_stem, toc(bench_timer) / 60);

function metric = score_reconstruction(clean, estimate)
clean = double(clean);
estimate = double(estimate);
residual = estimate - clean;
correlation = corrcoef(clean(:), estimate(:));
relative_rmse = sqrt(mean(residual(:).^2)) / max(sqrt(mean(clean(:).^2)), 1e-12);
output_snr = 10 * log10(var(clean(:)) / max(var(residual(:)), 1e-12));
metric = [correlation(1, 2), relative_rmse, output_snr];
end

function output = denoise_with_custom_avgref(chanlocs, srate, data, refCOV)
EEG = make_eeg_struct(chanlocs, srate, data);
EEG_clean = GEDAI(EEG, 'auto', 12, 0.5, refCOV, false, false, ...
    inf, inf, 'eeg', Inf, 'AvgRef', struct('silent', true));
output = double(EEG_clean.data);
end

function output = denoise_with_precomputed(chanlocs, srate, data)
EEG = make_eeg_struct(chanlocs, srate, data);
EEG_clean = GEDAI(EEG, 'auto', 12, 0.5, 'precomputed', false, false, ...
    inf, inf, 'eeg', Inf, 'AvgRef', struct('silent', true));
output = double(EEG_clean.data);
end

function output = denoise_with_3d_refcov(chanlocs, srate, data, refCOV_3D, R_np1)
EEG = make_eeg_struct(chanlocs, srate, R_np1 * data);
EEG.ref = 'average';
EEG_clean = GEDAI(EEG, 'auto', 12, 0.5, refCOV_3D, false, false, ...
    inf, inf, 'eeg', Inf, 'None', struct('silent', true));
output = double(EEG_clean.data);
end

function EEG = make_eeg_struct(chanlocs, srate, data)
EEG = struct();
EEG.setname = 'bear_semiblind_test_segment';
EEG.data = data;
EEG.srate = srate;
EEG.chanlocs = chanlocs;
EEG.nbchan = size(data, 1);
EEG.pnts = size(data, 2);
EEG.trials = 1;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.etc = struct();
end

function print_benchmark_summary(title_str, metrics_sub, clean_idx, cat_labels, methods, categories)
num_trials = size(metrics_sub, 1);
fprintf('\n====================================================================\n');
fprintf('%s (N = %d held-out 30s segments)\n', title_str, num_trials);
fprintf('====================================================================\n');
for m = 1:numel(methods)
    fprintf('%-26s Corr %.4f | RRMSE %.4f | SNR %.3f dB\n', methods{m}, ...
        mean(metrics_sub(:, m, 1)), mean(metrics_sub(:, m, 2)), mean(metrics_sub(:, m, 3)));
end

for base_idx = [2, 3]
    fprintf('\n--- Comparison vs %s ---\n', methods{base_idx});
    metric_names = {'Correlation', 'RRMSE', 'SNR (dB)'};
    for mi = [3, 2, 1]
        delta = metrics_sub(:, 4, mi) - metrics_sub(:, base_idx, mi);
        [~, p_t, ci] = ttest(delta);
        p_sr = signrank(delta);
        if mi == 2
            wins = sum(delta < 0);
        else
            wins = sum(delta > 0);
        end
        fprintf('  %-12s delta: mean=%+.4f, 95%% CI=[%+.4f,%+.4f], wins=%d/%d, t-p=%.4g, sr-p=%.4g\n', ...
            metric_names{mi}, mean(delta), ci(1), ci(2), wins, num_trials, p_t, p_sr);
    end
    delta_snr = metrics_sub(:, 4, 3) - metrics_sub(:, base_idx, 3);
    subj_deltas = zeros(10, 1);
    for s = 1:10
        subj_deltas(s) = mean(delta_snr(clean_idx == s));
    end
    [~, p_subj_t, ci_subj] = ttest(subj_deltas);
    p_subj_sr = signrank(subj_deltas);
    fprintf('  Subject-clustered SNR (N=10): wins=%d/10, 95%% CI=[%+.4f,%+.4f], t-p=%.4g, sr-p=%.4g\n', ...
        sum(subj_deltas > 0), ci_subj(1), ci_subj(2), p_subj_t, p_subj_sr);
    for c = 1:numel(categories)
        in_cat = cat_labels == string(categories{c});
        d_cat = delta_snr(in_cat);
        fprintf('  %-16s SNR delta: mean=%+.4f dB, wins=%d/%d\n', ...
            categories{c}, mean(d_cat), sum(d_cat > 0), numel(d_cat));
    end
end
end
