function [best_strength, best_SENSAI, opt_results] = GEDAI_bayesopt_strength(data, srate, chanlocs, ref_matrix, max_evals, verbose, plot_opt, signal_type, epoch_size_in_cycles, lowcut_frequency, parallelize)
% GEDAI_BAYESOPT_STRENGTH - Coarse-bracketed fminbnd loop to find the optimal
% deartifacting strength in range [0, 10] that maximizes the SENSAI score.
%
% Inputs:
%   data                 - (nbchan x pnts) double EEG sensor data
%   srate                - Sampling rate in Hz
%   chanlocs             - EEG.chanlocs struct
%   ref_matrix           - Precomputed string ('precomputed', 'interpolated', 'warped')
%                          or numeric reference covariance matrix (nbchan x nbchan)
%   max_evals            - Maximum objective evaluations (default = 12)
%   verbose              - 1 to print log, 0 for quiet (default = 1)
%   plot_opt             - Boolean to show optimization progress plot (default = false)
%   signal_type          - 'eeg' or 'meg' (default = 'eeg')
%   epoch_size_in_cycles - Epoch size in cycles (default = 12)
%   lowcut_frequency     - Low-cut frequency in Hz (default = 0.5)
%   parallelize          - Boolean to run each GEDAI evaluation in parallel (default = false)
%
% Outputs:
%   best_strength        - Optimal artifacting strength in [0, 10]
%   best_SENSAI          - Maximal SENSAI score (%) achieved
%   opt_results          - Optimization results struct

if nargin < 5 || isempty(max_evals), max_evals = 12; end
if nargin < 6 || isempty(verbose), verbose = 1; end
if nargin < 7 || isempty(plot_opt), plot_opt = false; end
if nargin < 8 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 9 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 10 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 11 || isempty(parallelize), parallelize = false; end
parallelize = logical(parallelize);

% Ensure data is double
data = double(data);

% Limit evaluation data to 60 seconds for fast convergence
max_samples = round(60 * srate);
if size(data, 2) > max_samples
    eval_data = data(:, 1:max_samples);
else
    eval_data = data;
end

% Build evaluation EEG structure
EEG_eval = struct();
EEG_eval.data = eval_data;
EEG_eval.srate = srate;
EEG_eval.chanlocs = chanlocs;
EEG_eval.nbchan = size(eval_data, 1);
EEG_eval.pnts = size(eval_data, 2);
EEG_eval.trials = 1;
EEG_eval.xmin = 0;
EEG_eval.xmax = (EEG_eval.pnts - 1) / srate;
EEG_eval.etc = struct();

% Coarse anchor points across active range [3.5, 10.0]
% (for strength <= 3.5, T1 = (105 - strength)/100 >= 1.015 removes 0 components)
seed_strengths = [3.5; 5.5; 7.0; 8.2; 9.1; 9.8];
if max_evals < numel(seed_strengths)
    seed_strengths = linspace(4.0, 9.8, max(2, max_evals)).';
end

if verbose
    fprintf('\n=== GEDAI: fminbnd Optimization of Artifacting Strength ===\n');
    fprintf('  Evaluating deartifacting strength in [3.5, 10] (Parallel = %d)\n', double(parallelize));
    fprintf('  Objective: Maximize SENSAI score (Coarse bracket + fminbnd refinement)...\n');
end

% Suppress warnings during optimization
warn_state = warning('off');
clean_up = onCleanup(@() warning(warn_state));

eval_hist_x = [];
eval_hist_f = [];

    function neg_s = tracked_obj(s_val)
        neg_s = eval_strength_objective(s_val, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize);
        eval_hist_x(end+1, 1) = s_val;
        eval_hist_f(end+1, 1) = neg_s;
    end

% Step 1: Evaluate coarse anchor points to bracket the active global peak
n_seeds = numel(seed_strengths);
seed_f = zeros(n_seeds, 1);
for k = 1:n_seeds
    seed_f(k) = tracked_obj(seed_strengths(k));
end

[~, idx_best] = min(seed_f);
a_sub = seed_strengths(max(1, idx_best - 1));
if idx_best == n_seeds
    b_sub = 10.0;
else
    b_sub = seed_strengths(idx_best + 1);
end

% Step 2: Refine within the winning bracket [a_sub, b_sub] using fminbnd
rem_evals = max(0, max_evals - n_seeds);
if rem_evals >= 2 && (b_sub - a_sub) > 0.1
    opts = optimset('TolX', 0.1, 'MaxFunEvals', rem_evals, 'Display', 'off');
    fminbnd(@tracked_obj, a_sub, b_sub, opts);
end

[min_neg_sensai, best_idx] = min(eval_hist_f);
best_strength = eval_hist_x(best_idx);
best_SENSAI = -min_neg_sensai;

opt_results = struct();
opt_results.XAtMinObjective = struct('strength', best_strength);
opt_results.MinObjective = min_neg_sensai;
opt_results.EvaluatedStrengths = eval_hist_x;
opt_results.EvaluatedSENSAI = -eval_hist_f;
opt_results.NumEvaluations = numel(eval_hist_x);

if plot_opt
    figure('Color', 'w', 'Name', 'GEDAI Strength Optimization (fminbnd)');
    [xs, ord] = sort(eval_hist_x);
    ys = -eval_hist_f(ord);
    plot(xs, ys, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;
    plot(best_strength, best_SENSAI, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xlabel('Artifacting Strength');
    ylabel('SENSAI Score (%)');
    title(sprintf('Optimal Strength = %.3f (SENSAI = %.2f%%)', best_strength, best_SENSAI));
    grid on;
end

if verbose
    fprintf('  Optimal Artifacting Strength: strength* = %.3f (%d evals)\n', best_strength, numel(eval_hist_x));
    fprintf('  Maximal SENSAI Score:         %+.2f %%\n', best_SENSAI);
    fprintf('===========================================================\n\n');
end

end

function neg_sensai = eval_strength_objective(strength_val, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize)
    try
        [~, ~, sensai] = GEDAI(EEG_eval, strength_val, epoch_size_in_cycles, lowcut_frequency, ...
            ref_matrix, parallelize, false, Inf, Inf, signal_type, Inf, '', ...
            'prior_blend', 'off', struct('silent', true));
        neg_sensai = -sensai;
    catch
        neg_sensai = 1e3; % Penalty on numerical failure
    end
end
