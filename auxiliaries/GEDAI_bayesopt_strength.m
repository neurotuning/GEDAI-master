function [best_scale, best_SENSAI, opt_results, selected_threshold_type] = GEDAI_bayesopt_strength(data, srate, chanlocs, ref_matrix, max_evals, verbose, plot_opt, signal_type, epoch_size_in_cycles, lowcut_frequency, parallelize)
% GEDAI_BAYESOPT_STRENGTH - Bayesian Optimization (bayesopt) over the auto_scale
% denoising multiplier in [0.1, 5.0] (from 0.1x most aggressive to 5.0x least
% aggressive relative to 'auto' = 1.0x), preserving per-band Z3 spectral-spatial
% adaptation across Pass 1 (Broadband) and Pass 2 (Wavelets).
%
% Inputs:
%   data                    - (nbchan x pnts) double EEG/MEG sensor data
%   srate                   - Sampling rate in Hz
%   chanlocs                - EEG.chanlocs struct
%   ref_matrix              - Precomputed string ('auto', 'precomputed', 'interpolated', ' projection')
%                             or numeric reference covariance matrix (nbchan x nbchan)
%   max_evals               - Maximum objective evaluations (default = 12)
%   verbose                 - 1 to print log, 0 for quiet (default = 1)
%   plot_opt                - Boolean to show optimization progress plot (default = false)
%   signal_type             - 'eeg' or 'meg' (default = 'eeg')
%   epoch_size_in_cycles    - Epoch size in cycles (default = 12)
%   lowcut_frequency        - Low-cut frequency in Hz (default = 0.5)
%   parallelize             - Boolean to run each GEDAI evaluation in parallel (default = false)
%
% Outputs:
%   best_scale              - Optimal auto_scale multiplier in [0.1, 5.0] (1.0 = 'auto')
%   best_SENSAI             - SENSAI score (%) at the selected multiplier
%   opt_results             - Optimization results struct (including bayesopt object)
%   selected_threshold_type - 'auto:<best_scale>' if it beats 'auto', else 'auto'

if nargin < 5 || isempty(max_evals), max_evals = 12; end
if nargin < 6 || isempty(verbose), verbose = 1; end
if nargin < 7 || isempty(plot_opt), plot_opt = false; end
if nargin < 8 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 9 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 10 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 11 || isempty(parallelize), parallelize = false; end

data = double(data);
[nbchan, pnts] = size(data);

% Subsample up to 60 seconds using 3 representative blocks (early/mid/late)
% with cosine-tapered seams for fast evaluation on long recordings
max_opt_sec = 60;
max_opt_pnts = min(pnts, round(max_opt_sec * srate));
if pnts > round(max_opt_pnts * 1.25)
    blk_len = floor(max_opt_pnts / 3);
    idx1 = 1:blk_len;
    mid_start = max(1, round(pnts / 2) - floor(blk_len / 2));
    idx2 = mid_start:(mid_start + blk_len - 1);
    idx3 = (pnts - blk_len + 1):pnts;
    data_sub = stitch_blocks_cosine(data(:, idx1), data(:, idx2), data(:, idx3), min(round(0.25 * srate), floor(blk_len / 8)));
else
    data_sub = data(:, 1:max_opt_pnts);
end

EEG_eval = struct();
EEG_eval.data = data_sub;
EEG_eval.srate = srate;
EEG_eval.chanlocs = chanlocs;
EEG_eval.nbchan = nbchan;
EEG_eval.pnts = size(data_sub, 2);
EEG_eval.trials = 1;
EEG_eval.xmin = 0;
EEG_eval.xmax = (EEG_eval.pnts - 1) / srate;
EEG_eval.etc = struct();

if verbose
    fprintf('\n=== GEDAI: Bayesian Optimization (bayesopt) of Denoising Strength ===\n');
    fprintf('  Sweeping auto_scale multiplier in [0.10x (most aggressive) .. 5.00x (least aggressive)]\n');
    fprintf('  Baseline anchor: ''auto'' (1.00x) | ''auto+'' (0.33x) | ''auto-'' (3.00x)\n');
end

warn_state = warning('off');
clean_up = onCleanup(@() warning(warn_state));

% Step 0: Evaluate 'auto' (1.0x) baseline first for adaptive ENOVA ceiling and monotonic guarantee
[sensai_auto, en_auto] = eval_autoscale_metrics(1.0, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize);
if en_auto < 18.0
    en_cap = en_auto + 1.5;
else
    en_cap = max(52.0, en_auto);
end

eval_hist_scale  = 1.0;
eval_hist_sensai = sensai_auto;
eval_hist_enova  = en_auto;
eval_hist_obj    = sensai_auto; % At 1.0x, ENOVA penalty is 0

% Initial log-spaced anchor seeds across [0.10, 5.00] including 0.10x, 0.25x, 0.33x (auto+), 1.0x (auto), 3.0x (auto-), 5.0x
default_seeds = [0.10; 0.18; 1/3; 0.60; 1.50; 3.00; 5.00];
n_init = min(numel(default_seeds), max(1, max_evals - 1));
init_scales = default_seeds(1:n_init);

rem_budget = max(1, max_evals - 1);
use_bayesopt = ( exist('bayesopt', 'file') == 2 || exist('bayesopt', 'class') == 8 ) && (rem_budget >= 4);

if use_bayesopt
    scale_var = optimizableVariable('auto_scale', [0.10, 5.00], 'Type', 'real', 'Transform', 'log');
    init_table = table(init_scales, 'VariableNames', {'auto_scale'});
    bo_obj = @(T) bayesopt_tracked_objective(T.auto_scale);
    try
        bo_res = bayesopt(bo_obj, scale_var, ...
            'MaxObjectiveEvaluations', rem_budget, ...
            'InitialX', init_table, ...
            'IsObjectiveDeterministic', true, ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'Verbose', 0, ...
            'PlotFcn', {});
    catch
        use_bayesopt = false;
    end
end

if ~use_bayesopt
    % Fallback if Statistics and Machine Learning Toolbox bayesopt is unavailable
    for k = 1:numel(init_scales)
        bayesopt_tracked_objective(init_scales(k));
    end
    [~, idx_b] = max(eval_hist_obj);
    sc_center = eval_hist_scale(idx_b);
    sc_lo = max(0.10, sc_center * 0.60);
    sc_hi = min(5.00, sc_center * 1.65);
    rem_f = max(0, max_evals - numel(eval_hist_scale));
    if rem_f >= 2
        opts = optimset('TolX', 0.02, 'MaxFunEvals', rem_f, 'Display', 'off');
        fminbnd(@(log_sc) bayesopt_tracked_objective(exp(log_sc)), log(sc_lo), log(sc_hi), opts);
    end
    bo_res = [];
end

[best_obj_val, best_idx] = max(eval_hist_obj);
best_scale  = eval_hist_scale(best_idx);
best_SENSAI = eval_hist_sensai(best_idx);
best_ENOVA  = eval_hist_enova(best_idx);

% Monotonic baseline guarantee: if no candidate strictly beats 'auto' (1.0x), retain 'auto'
if best_obj_val > sensai_auto + 1e-6 && abs(best_scale - 1.0) > 1e-3
    selected_threshold_type = sprintf('auto:%.6f', best_scale);
else
    best_scale  = 1.0;
    best_SENSAI = sensai_auto;
    best_ENOVA  = en_auto;
    selected_threshold_type = 'auto';
end

opt_results = struct();
opt_results.OptimalAutoScale      = best_scale;
opt_results.BestSENSAI            = best_SENSAI;
opt_results.BestENOVA             = best_ENOVA;
opt_results.BaselineAutoSENSAI    = sensai_auto;
opt_results.BaselineAutoENOVA     = en_auto;
opt_results.ENOVACeiling          = en_cap;
opt_results.SelectedThresholdType = selected_threshold_type;
opt_results.EvaluatedScales       = eval_hist_scale;
opt_results.EvaluatedSENSAI       = eval_hist_sensai;
opt_results.EvaluatedENOVA        = eval_hist_enova;
opt_results.EvaluatedObjective    = eval_hist_obj;
opt_results.NumEvaluations        = numel(eval_hist_scale);
opt_results.BayesOptResults       = bo_res;

if plot_opt
    figure('Color', 'w', 'Name', 'GEDAI Strength Optimization (bayesopt 0.1x - 5.0x)');
    [xs, ord] = sort(eval_hist_scale);
    ys = eval_hist_obj(ord);
    semilogx(xs, ys, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;
    yline(sensai_auto, '--k', sprintf('''auto'' (1.0x) baseline (%.2f%%)', sensai_auto), 'LineWidth', 1.2);
    semilogx(best_scale, best_obj_val, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xlabel('Denoising Multiplier auto\_scale (0.1x = most aggressive, 5.0x = least aggressive)');
    ylabel('Regularized SENSAI Objective (%)');
    title(sprintf('Optimal auto\\_scale = %.2fx (SENSAI = %.2f%%, ENOVA = %.1f%%)', best_scale, best_SENSAI, best_ENOVA));
    grid on;
end

if verbose
    fprintf('  Baseline ''auto'' (1.00x):      SENSAI = %+.2f%% | ENOVA = %.1f%% (Ceiling = %.1f%%)\n', sensai_auto, en_auto, en_cap);
    fprintf('  Selected Optimal Multiplier:  auto_scale* = %.2fx (%d evals)\n', best_scale, opt_results.NumEvaluations);
    fprintf('  Selected SENSAI / ENOVA:      SENSAI = %+.2f%% | ENOVA = %.1f%%\n', best_SENSAI, best_ENOVA);
    fprintf('=====================================================================\n\n');
end

    function neg_obj = bayesopt_tracked_objective(sc_val)
        sc_val = max(0.10, min(5.00, double(sc_val)));
        % Reuse cached value if already evaluated within 1e-4
        [min_diff, idx_hit] = min(abs(eval_hist_scale - sc_val));
        if min_diff < 1e-4
            neg_obj = -eval_hist_obj(idx_hit);
            return;
        end
        [sen_v, en_v] = eval_autoscale_metrics(sc_val, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize);
        obj_v = sen_v - 2.0 * max(0, en_v - en_cap)^1.5;
        eval_hist_scale(end+1, 1)  = sc_val;
        eval_hist_sensai(end+1, 1) = sen_v;
        eval_hist_enova(end+1, 1)  = en_v;
        eval_hist_obj(end+1, 1)    = obj_v;
        neg_obj = -obj_v;
    end

end

function [sensai_score, enova_pct] = eval_autoscale_metrics(sc_val, EEG_in, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize)
try
    mode_str = sprintf('auto:%.6f', sc_val);
    [~, EEG_clean, EEG_art, sensai_score] = evalc( ...
        'GEDAI(EEG_in, mode_str, epoch_size_in_cycles, lowcut_frequency, ref_matrix, parallelize, false, inf, inf, signal_type, inf);');
    X_cl = double(EEG_clean.data);
    X_ar = double(EEG_art.data);
    X_in_hp = X_cl + X_ar;
    enova_pct = 100 * (1 - sum(X_cl(:).^2) / max(sum(X_in_hp(:).^2), 1e-15));
    if isnan(sensai_score) || isinf(sensai_score)
        sensai_score = -100;
        enova_pct = 100;
    end
catch
    sensai_score = -100;
    enova_pct = 100;
end
end

function out = stitch_blocks_cosine(b1, b2, b3, fade_len)
if fade_len < 2
    out = [b1, b2, b3];
    return;
end
w = 0.5 * (1 - cos(pi * (0:(fade_len - 1)) / (fade_len - 1)));
b12 = [b1(:, 1:end-fade_len), bsxfun(@times, b1(:, end-fade_len+1:end), 1 - w) + bsxfun(@times, b2(:, 1:fade_len), w), b2(:, fade_len+1:end)];
out = [b12(:, 1:end-fade_len), bsxfun(@times, b12(:, end-fade_len+1:end), 1 - w) + bsxfun(@times, b3(:, 1:fade_len), w), b3(:, fade_len+1:end)];
end

