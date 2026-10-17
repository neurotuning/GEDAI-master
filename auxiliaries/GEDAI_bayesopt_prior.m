function [best_gamma, best_SENSAI, C_optimal, opt_results] = GEDAI_bayesopt_prior(data, srate, chanlocs, C_LF, C_EC, max_evals, verbose, plot_opt)
% GEDAI_BAYESOPT_PRIOR - Outer Bayesian Optimization loop to find the optimal
% EWOK Riemannian blending parameter gamma in [0, 1] between the theoretical
% leadfield prior and empirical EC centroid that maximizes the SENSAI score.
%
% Inputs:
%   data       - (nbchan x pnts) double EEG sensor data
%   srate      - Sampling rate in Hz
%   chanlocs   - EEG.chanlocs struct
%   C_LF       - Theoretical leadfield covariance prior (nbchan x nbchan)
%   C_EC       - Empirical EC covariance centroid (nbchan x nbchan)
%   max_evals  - Maximum Bayesian optimization evaluations (default = 15)
%   verbose    - 1 to print log, 0 for quiet (default = 1)
%   plot_opt   - Boolean to show bayesopt progress plot (default = false)
%
% Outputs:
%   best_gamma  - Optimal blending parameter in [0, 1]
%   best_SENSAI - Maximal SENSAI score (%) achieved
%   C_optimal   - Optimal blended covariance matrix
%   opt_results - Full bayesopt results object

if nargin < 6 || isempty(max_evals), max_evals = 15; end
if nargin < 7 || isempty(verbose), verbose = 1; end
if nargin < 8 || isempty(plot_opt), plot_opt = false; end

% Ensure data is double
data = double(data);

% Limit evaluation data to 60 seconds for ultra-fast surrogate convergence
max_samples = round(60 * srate);
if size(data, 2) > max_samples
    eval_data = data(:, 1:max_samples);
else
    eval_data = data;
end

epoch_size = 1.0; % 1-second epochs for SENSAI evaluation
noise_mult = 1.0;

% Define objective function using true multi-band GEDAI on evaluation data
obj_fun = @(tbl) eval_objective(tbl.gamma, eval_data, srate, chanlocs, C_LF, C_EC);

% Optimization variable
opt_gamma = optimizableVariable('gamma', [0, 1], 'Type', 'real');

% Explicit seed points guaranteeing that gamma = 0.0 (leadfield / 'off') is evaluated
seed_gammas = [0.0; 0.25; 0.50; 0.75; 1.0];
if max_evals < 5
    seed_gammas = seed_gammas(1:max_evals);
end
initial_tbl = table(seed_gammas, 'VariableNames', {'gamma'});

if plot_opt
    plot_fcn = {@plotObjectiveModel};
else
    plot_fcn = {};
end

if verbose
    fprintf('\n=== GEDAI: Bayesian Optimization of Prior Geodesic Blending ===\n');
    fprintf('  Evaluating gamma in [0, 1] (0 = Leadfield, 1 = Empirical EC Centroid)\n');
    fprintf('  Objective: Maximize SENSAI score (Evaluating %d iterations)...\n', max_evals);
end

% Suppress warnings during optimization
warn_state = warning('off');
clean_up = onCleanup(@() warning(warn_state));

opt_results = bayesopt(obj_fun, opt_gamma, ...
    'InitialX', initial_tbl, ...
    'MaxObjectiveEvaluations', max_evals, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'IsObjectiveDeterministic', true, ...
    'PlotFcn', plot_fcn, ...
    'Verbose', 0);

best_gamma = opt_results.XAtMinObjective.gamma;
best_SENSAI = -opt_results.MinObjective;

% Compute final optimal blended covariance
C_optimal = GEDAI_ewok_geodesic_blend(C_LF, C_EC, best_gamma);

if verbose
    fprintf('  Optimal Blending Parameter: gamma* = %.4f\n', best_gamma);
    fprintf('  Maximal SENSAI Score:       %+.2f %%\n', best_SENSAI);
    fprintf('=================================================================\n\n');
end

end

function neg_sensai = eval_objective(gamma_val, data, srate, chanlocs, C_LF, C_EC)
    % 1. Blend reference covariance along geodesic
    C_trial = GEDAI_ewok_geodesic_blend(C_LF, C_EC, gamma_val);
    
    % 2. Evaluate true multi-band GEDAI with candidate covariance
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
        
        [~, ~, sensai] = GEDAI(EEG_eval, 'auto', 12, 0.5, C_trial, false, false, Inf, Inf, 'eeg', Inf, '', struct('silent', true));
        neg_sensai = -sensai;
    catch
        neg_sensai = 1e3; % Penalty on numerical failure
    end
end
