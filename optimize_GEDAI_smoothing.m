% [Generalized Eigenvalue De-Artifacting Instrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2025] Tomas Ros & Abele Michela - dr.t.ros@gmail.com

function [optimal_params, bayesopt_results] = optimize_GEDAI_smoothing(EEG, varargin)
%OPTIMIZE_GEDAI_SMOOTHING  Joint Bayesian Optimization of GEDAI hyperparameters.
%
%   Maximizes the global SENSAI score by jointly optimizing:
%     - smoothing_window_minutes  [0.01, 10]  (log scale)
%     - epoch_size_in_cycles      [4,   24]   (integer)
%
% Usage:
%   [params] = optimize_GEDAI_smoothing(EEG)
%   [params] = optimize_GEDAI_smoothing(EEG, 'MaxEvaluations', 20, ...)
%
% Required input:
%   EEG                     - EEG data struct in EEGlab format
%
% Optional name-value pairs:
%   'artifact_threshold'    - 'auto','auto+','auto-'. Default: 'auto'
%   'lowcut_frequency'      - Low-cut frequency Hz. Default: 0.5
%   'ref_matrix_type'       - Default: 'precomputed'
%   'signal_type'           - 'eeg' or 'meg'. Default: 'eeg'
%   'MaxEvaluations'        - Bayesopt iterations. Default: 20
%   'WindowRange'           - [min max] minutes. Default: [0.01 10]
%   'EpochCyclesRange'      - [min max] integer cycles. Default: [4 24]
%
% Outputs:
%   optimal_params          - struct with fields:
%                               .smoothing_window_minutes
%                               .epoch_size_in_cycles
%                               .SENSAI_score
%   bayesopt_results        - full bayesopt result object

p = inputParser;
addParameter(p, 'artifact_threshold',  'auto');
addParameter(p, 'lowcut_frequency',    0.5);
addParameter(p, 'ref_matrix_type',     'precomputed');
addParameter(p, 'signal_type',         'eeg');
addParameter(p, 'MaxEvaluations',      20);
addParameter(p, 'WindowRange',         [0.01, 10]);
addParameter(p, 'EpochCyclesRange',    [2, 24]);
addParameter(p, 'parallel',            true);
p.parse(varargin{:});
opts = p.Results;

% Auto-set the upper window bound to the full recording length (in minutes)
% so that WindowRange(2) = Inf (whole file) is always reachable.
recording_minutes = EEG.pnts / EEG.srate / 60;
if opts.WindowRange(2) > recording_minutes
    opts.WindowRange(2) = recording_minutes;
end
% Ensure lower bound doesn't exceed upper bound
opts.WindowRange(1) = min(opts.WindowRange(1), opts.WindowRange(2));

fprintf('\n=== optimize_GEDAI_smoothing: Joint Bayesian Optimization ===\n');
fprintf('Parameters: smoothing_window_minutes in [%.3f, %.1f] min  |  epoch_size_in_cycles in [%d, %d]\n', ...
    opts.WindowRange(1), opts.WindowRange(2), opts.EpochCyclesRange(1), opts.EpochCyclesRange(2));
fprintf('MaxEvaluations = %d\n\n', opts.MaxEvaluations);

% --- Define optimizable variables ---
x_window = optimizableVariable('window_minutes', opts.WindowRange, ...
    'Type', 'real', 'Transform', 'log');

x_cycles = optimizableVariable('epoch_cycles', opts.EpochCyclesRange, ...
    'Type', 'integer');

% --- Objective function ---
obj_fn = @(x) gedai_joint_objective(EEG, x.window_minutes, x.epoch_cycles, opts);

% --- Run Bayesian Optimization ---
bayesopt_results = bayesopt(obj_fn, [x_window, x_cycles], ...
    'MaxObjectiveEvaluations', opts.MaxEvaluations, ...
    'IsObjectiveDeterministic',       false, ...
    'AcquisitionFunctionName',        'expected-improvement-plus', ...
    'ExplorationRatio',               0.5, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
    'Verbose', 1);

% --- Collect results ---
optimal_params.smoothing_window_minutes = bayesopt_results.XAtMinObjective.window_minutes;
optimal_params.epoch_size_in_cycles     = bayesopt_results.XAtMinObjective.epoch_cycles;
optimal_params.SENSAI_score             = -bayesopt_results.MinObjective;

fprintf('\n=== Optimization Complete ===\n');
fprintf('Optimal smoothing window   : %.3f minutes (%.1f seconds)\n', ...
    optimal_params.smoothing_window_minutes, optimal_params.smoothing_window_minutes * 60);
fprintf('Optimal epoch size         : %d cycles\n',  optimal_params.epoch_size_in_cycles);
fprintf('Best SENSAI score          : %.4f%%\n\n',   optimal_params.SENSAI_score);
end


% =========================================================================
% Local objective function
% =========================================================================
function neg_score = gedai_joint_objective(EEG, window_minutes, epoch_cycles, opts)

try
    [~, ~, SENSAI_score] = GEDAI(EEG, ...
        opts.artifact_threshold, ...
        epoch_cycles, ...           % epoch_size_in_cycles  (optimized)
        opts.lowcut_frequency, ...
        opts.ref_matrix_type, ...
        opts.parallel, ...          % parallel
        false, ...                  % visualize_artifacts — off
        inf,  ...                   % ENOVA_threshold: no rejection
        opts.signal_type, ...
        false, ...                  % visualize_manifold — off
        window_minutes);            % smoothing_window_minutes (optimized)

    neg_score = -SENSAI_score;
    fprintf('  [eval] window=%.3f min  cycles=%d  →  SENSAI=%.4f\n', ...
        window_minutes, epoch_cycles, SENSAI_score);

catch ME
    warning('optimize_GEDAI_smoothing: GEDAI failed (window=%.3f min, cycles=%d).\n  Reason: %s', ...
        window_minutes, epoch_cycles, ME.message);
    neg_score = 0;
end
end
