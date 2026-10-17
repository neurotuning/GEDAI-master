function [best_strength, best_SENSAI, opt_results] = GEDAI_bayesopt_strength(data, srate, chanlocs, ref_matrix, max_evals, verbose, plot_opt, signal_type, epoch_size_in_cycles, lowcut_frequency)
% GEDAI_BAYESOPT_STRENGTH - Bayesian Optimization loop to find the optimal
% deartifacting strength in range [0, 10] that maximizes the SENSAI score.
%
% Inputs:
%   data                 - (nbchan x pnts) double EEG sensor data
%   srate                - Sampling rate in Hz
%   chanlocs             - EEG.chanlocs struct
%   ref_matrix           - Precomputed string ('precomputed', 'interpolated', 'warped')
%                          or numeric reference covariance matrix (nbchan x nbchan)
%   max_evals            - Maximum Bayesian optimization evaluations (default = 15)
%   verbose              - 1 to print log, 0 for quiet (default = 1)
%   plot_opt             - Boolean to show bayesopt progress plot (default = false)
%   signal_type          - 'eeg' or 'meg' (default = 'eeg')
%   epoch_size_in_cycles - Epoch size in cycles (default = 12)
%   lowcut_frequency     - Low-cut frequency in Hz (default = 0.5)
%
% Outputs:
%   best_strength        - Optimal artifacting strength in [0, 10]
%   best_SENSAI          - Maximal SENSAI score (%) achieved
%   opt_results          - Full bayesopt results object

if nargin < 5 || isempty(max_evals), max_evals = 15; end
if nargin < 6 || isempty(verbose), verbose = 1; end
if nargin < 7 || isempty(plot_opt), plot_opt = false; end
if nargin < 8 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 9 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 10 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end

% Ensure data is double
data = double(data);

% Limit evaluation data to 60 seconds for ultra-fast surrogate convergence
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

% Define objective function maximizing SENSAI score
obj_fun = @(tbl) eval_strength_objective(tbl.strength, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type);

% Optimization variable: strength in [0, 10]
opt_strength = optimizableVariable('strength', [0, 10], 'Type', 'real');

% Explicit seed points covering standard settings:
% 7.0 ('auto'), 8.5 ('auto+'), 4.0 ('auto-'), 0.0 (minimal/off), 2.0 (mild), 9.0, 10.0 (maximum)
seed_strengths = [7.0; 8.5; 4.0; 0.0; 2.0; 9.0; 10.0];
if max_evals < length(seed_strengths)
    seed_strengths = seed_strengths(1:max_evals);
end
initial_tbl = table(seed_strengths, 'VariableNames', {'strength'});

if plot_opt
    plot_fcn = {@plotObjectiveModel};
else
    plot_fcn = {};
end

if verbose
    fprintf('\n=== GEDAI: Bayesian Optimization of Artifacting Strength ===\n');
    fprintf('  Evaluating deartifacting strength in [0, 10] (0 = Mildest, 7 = Auto, 10 = Strongest)\n');
    fprintf('  Objective: Maximize SENSAI score (Evaluating %d iterations)...\n', max_evals);
end

% Suppress warnings during optimization
warn_state = warning('off');
clean_up = onCleanup(@() warning(warn_state));

opt_results = bayesopt(obj_fun, opt_strength, ...
    'InitialX', initial_tbl, ...
    'MaxObjectiveEvaluations', max_evals, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'IsObjectiveDeterministic', true, ...
    'PlotFcn', plot_fcn, ...
    'Verbose', 0);

best_strength = opt_results.XAtMinObjective.strength;
best_SENSAI = -opt_results.MinObjective;

if verbose
    fprintf('  Optimal Artifacting Strength: strength* = %.3f\n', best_strength);
    fprintf('  Maximal SENSAI Score:         %+.2f %%\n', best_SENSAI);
    fprintf('============================================================\n\n');
end

end

function neg_sensai = eval_strength_objective(strength_val, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type)
    try
        [~, ~, sensai] = GEDAI(EEG_eval, strength_val, epoch_size_in_cycles, lowcut_frequency, ...
            ref_matrix, false, false, Inf, Inf, signal_type, Inf, '', ...
            'prior_blend', 'off', struct('silent', true));
        neg_sensai = -sensai;
    catch
        neg_sensai = 1e3; % Penalty on numerical failure
    end
end
