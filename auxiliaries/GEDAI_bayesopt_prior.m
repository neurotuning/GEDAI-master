function [best_gamma, best_SENSAI, C_optimal, opt_results] = GEDAI_bayesopt_prior(...
    data, srate, chanlocs, C_LF, C_EC, max_evals, verbose, plot_opt, ...
    artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, signal_type, smoothing_window_seconds, output_reference_channel, parallelize)
% GEDAI_BAYESOPT_PRIOR - Coarse-bracketed fminbnd loop to find the optimal
% EWOK Riemannian blending parameter gamma in [0, 1] between the theoretical
% leadfield prior (C_LF) and empirical EC centroid (C_EC) that maximizes
% the SENSAI score evaluated against the candidate blended prior (C_trial).
%
% Inputs:
%   data                     - (nbchan x pnts) double EEG sensor data
%   srate                    - Sampling rate in Hz
%   chanlocs                 - EEG.chanlocs struct
%   C_LF                     - Theoretical leadfield covariance prior (nbchan x nbchan)
%   C_EC                     - Empirical EC covariance centroid (nbchan x nbchan)
%   max_evals                - Maximum objective evaluations (default = 11)
%   verbose                  - 1 to print log, 0 for quiet (default = 1)
%   plot_opt                 - Boolean to show optimization progress plot (default = false)
%   artifact_threshold_type  - Threshold level ('auto', 'auto+', 'auto-', or numeric)
%   epoch_size_in_cycles     - Epoch size in cycles (default = 12)
%   lowcut_frequency         - Low-cut frequency in Hz (default = 0.5)
%   signal_type              - 'eeg' or 'meg' (default = 'eeg')
%   smoothing_window_seconds - Sliding window size in seconds (default = Inf)
%   output_reference_channel - Target montage reference channel (default = '')
%   parallelize              - Boolean to run each GEDAI evaluation in parallel (default = false)
%
% Outputs:
%   best_gamma  - Optimal blending parameter in [0, 1]
%   best_SENSAI - Maximal SENSAI score (%) achieved
%   C_optimal   - Optimal blended covariance matrix
%   opt_results - Optimization results struct

if nargin < 6 || isempty(max_evals), max_evals = 11; end
if nargin < 7 || isempty(verbose), verbose = 1; end
if nargin < 8 || isempty(plot_opt), plot_opt = false; end
if nargin < 9 || isempty(artifact_threshold_type), artifact_threshold_type = 'auto'; end
if nargin < 10 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 11 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 12 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 13 || isempty(smoothing_window_seconds), smoothing_window_seconds = Inf; end
if nargin < 14 || isempty(output_reference_channel), output_reference_channel = ''; end
if nargin < 15 || isempty(parallelize), parallelize = false; end
parallelize = logical(parallelize);

% Ensure data is double
data = double(data);
pnts = size(data, 2);

% Subsample evaluation data: use full dataset if <= 180s (3 minutes),
% or 3 representative 30-second segments (start, mid, end) for longer recordings
max_eval_pnts = round(180 * srate);
if pnts <= max_eval_pnts
    eval_data = data;
else
    seg_len = round(30 * srate);
    idx1 = 1:seg_len;
    mid_start = round(pnts/2 - seg_len/2);
    idx2 = mid_start:(mid_start + seg_len - 1);
    idx3 = (pnts - seg_len + 1):pnts;
    eval_data = [data(:, idx1), data(:, idx2), data(:, idx3)];
end

% Explicit seed points guaranteeing that gamma = 0.0 (pure leadfield baseline) is evaluated first
seed_gammas = [0.0; 0.25; 0.50; 0.75; 1.0];
if max_evals < 5
    seed_gammas = seed_gammas(1:max(1, max_evals));
end

if verbose
    fprintf('\n=== GEDAI: fminbnd Optimization of Prior Geodesic Blending ===\n');
    fprintf('  Evaluating gamma in [0, 1] (0 = Leadfield, 1 = Empirical EC Centroid | Parallel = %d)\n', double(parallelize));
    fprintf('  Objective: Maximize SENSAI score vs candidate blended prior...\n');
end

% Suppress warnings during optimization
warn_state = warning('off');
clean_up = onCleanup(@() warning(warn_state));

eval_hist_g = [];
eval_hist_f = [];

    function neg_s = tracked_prior_obj(g_val)
        neg_s = eval_objective(g_val, eval_data, srate, chanlocs, C_LF, C_EC, ...
            artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, signal_type, smoothing_window_seconds, output_reference_channel, parallelize);
        eval_hist_g(end+1, 1) = g_val;
        eval_hist_f(end+1, 1) = neg_s;
    end

n_seeds = numel(seed_gammas);
seed_f = zeros(n_seeds, 1);
for k = 1:n_seeds
    seed_f(k) = tracked_prior_obj(seed_gammas(k));
end
baseline_sensai = -seed_f(1);

[~, idx_best] = min(seed_f);
a_sub = seed_gammas(max(1, idx_best - 1));
b_sub = seed_gammas(min(n_seeds, idx_best + 1));

rem_evals = max(0, max_evals - n_seeds);
if rem_evals >= 2 && (b_sub - a_sub) > 0.02
    opts = optimset('TolX', 0.01, 'MaxFunEvals', rem_evals, 'Display', 'off');
    fminbnd(@tracked_prior_obj, a_sub, b_sub, opts);
end

[min_neg_sensai, best_idx] = min(eval_hist_f);
best_gamma = eval_hist_g(best_idx);
best_SENSAI = -min_neg_sensai;

% Monotonic baseline guarantee: If blending does not outperform baseline leadfield, retain gamma = 0.0
if best_SENSAI <= baseline_sensai
    best_gamma = 0.0;
    best_SENSAI = baseline_sensai;
    C_optimal = C_LF;
else
    C_optimal = GEDAI_ewok_geodesic_blend(C_LF, C_EC, best_gamma);
end

opt_results = struct();
opt_results.XAtMinObjective = struct('gamma', best_gamma);
opt_results.MinObjective = -best_SENSAI;
opt_results.EvaluatedGammas = eval_hist_g;
opt_results.EvaluatedSENSAI = -eval_hist_f;
opt_results.NumEvaluations = numel(eval_hist_g);

if plot_opt
    figure('Color', 'w', 'Name', 'GEDAI Prior Blending Optimization (fminbnd)');
    [gs, ord] = sort(eval_hist_g);
    ys = -eval_hist_f(ord);
    plot(gs, ys, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;
    plot(best_gamma, best_SENSAI, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xlabel('Blending Parameter \gamma');
    ylabel('SENSAI Score (%)');
    title(sprintf('Optimal \\gamma = %.4f (SENSAI = %.2f%%)', best_gamma, best_SENSAI));
    grid on;
end

if verbose
    fprintf('  Baseline Leadfield SENSAI:  %+.2f %%\n', baseline_sensai);
    fprintf('  Optimal Blending Parameter: gamma* = %.4f (%d evals)\n', best_gamma, numel(eval_hist_g));
    fprintf('  Maximal SENSAI Score:       %+.2f %%\n', best_SENSAI);
    fprintf('==============================================================\n\n');
end

end

function neg_sensai = eval_objective(gamma_val, data, srate, chanlocs, C_LF, C_EC, ...
    artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, signal_type, smoothing_window_seconds, output_reference_channel, parallelize)

    % 1. Blend reference covariance along geodesic
    C_trial = GEDAI_ewok_geodesic_blend(C_LF, C_EC, gamma_val);
    
    % 2. Clean evaluation data with candidate covariance
    try
        EEG_eval = struct();
        EEG_eval.data = data;
        EEG_eval.srate = srate;
        EEG_eval.chanlocs = chanlocs;
        EEG_eval.nbchan = size(data, 1);
        EEG_eval.pnts = size(data, 2);
        EEG_eval.trials = 1;
        EEG_eval.xmin = 0;
        EEG_eval.xmax = (EEG_eval.pnts - 1) / srate;
        EEG_eval.etc = struct();
        
        pass_opts = struct('silent', true);
        [EEG_clean_eval, EEG_art_eval] = GEDAI(EEG_eval, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ...
            C_trial, parallelize, false, Inf, Inf, signal_type, smoothing_window_seconds, output_reference_channel, ...
            'prior_blend', 'off', pass_opts);
        
        % 3. Evaluate SENSAI against the candidate blended reference C_trial
        sensai = SENSAI_basic(double(EEG_clean_eval.data), double(EEG_art_eval.data), srate, 1, C_trial, 1.0, signal_type);
        neg_sensai = -sensai;
    catch
        neg_sensai = 1e3; % Penalty on numerical failure
    end
end
