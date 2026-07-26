function [best_params, best_EEGclean, results] = GEDAI_bayesopt(EEGin, varargin)
% GEDAI_BAYESOPT Bayesian Optimization of GEDAI Hyperparameters
%
% Usage:
%   [best_params, best_EEGclean, results] = GEDAI_bayesopt(EEGin)
%   [best_params, best_EEGclean, results] = GEDAI_bayesopt(EEGin, 'max_evals', 50, ...)
%
% Optimized Hyperparameters:
%   1. Denoising strength (1 to 9)
%   2. Epoch size in wave cycles (2 to 20)
%   3. Sliding window size in seconds (2 to 60, or Inf)
%
% Inputs:
%   EEGin                 - Input EEGLAB structure
%   'max_evals'           - Maximum number of objective evaluations (default: 20)
%   'num_seed_points'     - Number of initial random seed points (default: 5)
%   'lowcut_frequency'    - Low-cut frequency in Hz (default: 0.5)
%   'ref_matrix_type'     - Reference leadfield matrix type (default: 'precomputed')
%   'signal_type'         - Signal type: 'eeg' or 'meg' (default: 'eeg')
%   'parallel'            - Enable parallel processing (default: false)
%   'visualize'           - Enable realtime dark-themed visualization (default: true)
%   'denoising_range'     - Range for denoising strength [min, max] (default: [1, 9])
%   'epoch_cycles_range'  - Range for epoch size in wave cycles [min, max] (default: [2, 20])
%   'window_sec_range'    - Range for sliding window size in seconds [min, max] (default: [2, 61], >60 maps to Inf)
%   'verbose'             - Verbosity level for bayesopt (default: 0)
%
% Outputs:
%   best_params           - Struct with optimal hyperparameters and SENSAI score
%   best_EEGclean         - Cleaned EEGLAB structure using best hyperparameters
%   results               - MATLAB BayesianOptimization object
%
% NeuroTuning Lab - University of Geneva

%% 0. Handle Missing/Empty Input Dataset
if nargin < 1 || isempty(EEGin)
    if evalin('base', 'exist(''EEG'', ''var'')')
        EEGin = evalin('base', 'EEG');
        disp('Using ''EEG'' dataset from base workspace.');
    elseif evalin('base', 'exist(''ALLEEG'', ''var'')')
        ALLEEG_temp = evalin('base', 'ALLEEG');
        if ~isempty(ALLEEG_temp)
            EEGin = ALLEEG_temp(1);
            disp('Using first dataset from ''ALLEEG'' in base workspace.');
        end
    else
        if exist('pop_loadset', 'file')
            EEGin = pop_loadset();
        else
            [fname, fpath] = uigetfile('*.set', 'Select EEG dataset for GEDAI Bayesian Optimization');
            if isequal(fname, 0), return; end
            data_st = load(fullfile(fpath, fname), '-mat');
            if isfield(data_st, 'EEG'), EEGin = data_st.EEG; else, EEGin = data_st; end
        end
    end
end

%% 1. Parse Input Parameters
p = inputParser;
addRequired(p, 'EEGin', @isstruct);
addParameter(p, 'max_evals', 20, @isnumeric);
addParameter(p, 'num_seed_points', 5, @isnumeric);
addParameter(p, 'lowcut_frequency', 0.5, @isnumeric);
addParameter(p, 'ref_matrix_type', 'precomputed');
addParameter(p, 'signal_type', 'eeg', @(x) ischar(x) || isstring(x));
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'visualize', true, @islogical);
addParameter(p, 'denoising_range', [1, 9], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'epoch_cycles_range', [2, 20], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'smoothing_window_seconds', Inf, @isnumeric);
addParameter(p, 'default_denoising', 7, @isnumeric);
addParameter(p, 'default_epoch_cycles', 12, @isnumeric);
addParameter(p, 'verbose', 0, @isnumeric);

parse(p, EEGin, varargin{:});
opts = p.Results;

max_evals = opts.max_evals;
num_seed_points = opts.num_seed_points;
signal_type = char(opts.signal_type);

%% 2. Define Optimization Variables (2 Hyperparameters)
vars = [
    optimizableVariable('denoising_strength', opts.denoising_range, 'Type', 'real')
    optimizableVariable('epoch_size_in_cycles', opts.epoch_cycles_range, 'Type', 'integer')
];

%% 3. Initialize Optimization State & Realtime Visualization Setup
state = struct();
state.eval_count = 0;
state.max_evals = max_evals;
state.scores_history = [];
state.running_best_history = [];
state.best_score = -Inf;
state.best_eval_idx = 1;
state.best_params = struct();
state.best_thresholds = [];
state.best_EEGclean = [];
state.best_band_labels = {};
state.fig = [];

if opts.visualize
    state.fig = create_realtime_figure(max_evals);
end

%% 4. Define Objective Function Callback
    function objective = objective_func(X)
        state.eval_count = state.eval_count + 1;
        eval_idx = state.eval_count;
        
        d_strength = X.denoising_strength;
        epoch_cycles = round(X.epoch_size_in_cycles);
        win_sec_eval = opts.smoothing_window_seconds;
        
        try
            [EEGclean, EEGartifacts, sensai_val, ~, artifact_thresh_band, mean_enova, ~] = ...
                GEDAI(EEGin, d_strength, epoch_cycles, opts.lowcut_frequency, ...
                      opts.ref_matrix_type, opts.parallel, false, inf, inf, ...
                      signal_type, win_sec_eval, struct('silent', true));
            
            sensai_score = sensai_val;
            if isnan(sensai_score) || isinf(sensai_score)
                sensai_score = 0;
            end
        catch ME
            warning('GEDAI evaluation %d failed: %s', eval_idx, ME.message);
            sensai_score = 0;
            artifact_thresh_band = [];
            EEGclean = [];
        end
        
        objective = -sensai_score; % bayesopt minimizes objective
        
        % Update tracking state
        state.scores_history(eval_idx) = sensai_score;
        if eval_idx == 1
            current_max = sensai_score;
        else
            current_max = max(state.running_best_history(end), sensai_score);
        end
        state.running_best_history(eval_idx) = current_max;
        
        % Check if new best score found
        if sensai_score > state.best_score
            state.best_score = sensai_score;
            state.best_eval_idx = eval_idx;
            state.best_params = struct(...
                'denoising_strength', d_strength, ...
                'epoch_size_in_cycles', epoch_cycles, ...
                'SENSAI_score', sensai_score);
            state.best_thresholds = artifact_thresh_band;
            state.best_EEGclean = EEGclean;
            
            % Generate band frequency labels
            if ~isempty(EEGclean) && isfield(EEGclean.etc, 'GEDAI')
                gedai_info = EEGclean.etc.GEDAI;
                if isfield(gedai_info, 'upper_frequencies') && ~isempty(gedai_info.upper_frequencies)
                    freqs = gedai_info.upper_frequencies;
                elseif isfield(gedai_info, 'center_frequencies') && ~isempty(gedai_info.center_frequencies)
                    freqs = gedai_info.center_frequencies;
                else
                    freqs = [];
                end
                
                n_thresh = length(artifact_thresh_band);
                labels = cell(1, n_thresh);
                if ~isempty(freqs) && n_thresh == length(freqs) + 1
                    labels{1} = 'BB';
                    for k = 1:length(freqs)
                        val = freqs(k);
                        lbl = sprintf('%.1f', val);
                        labels{k+1} = strrep(lbl, '.0', '');
                    end
                elseif ~isempty(freqs) && n_thresh == length(freqs)
                    for k = 1:length(freqs)
                        val = freqs(k);
                        lbl = sprintf('%.1f', val);
                        labels{k} = strrep(lbl, '.0', '');
                    end
                else
                    for k = 1:n_thresh
                        labels{k} = sprintf('B%d', k);
                    end
                end
                state.best_band_labels = labels;
            end
        end
        
        % Update Realtime Visualization
        if opts.visualize && ishandle(state.fig)
            update_realtime_figure(state);
        end
    end

%% 5. Run Bayesian Optimization (Cold-started with default parameters as Evaluation #1)
initial_point = table(opts.default_denoising, opts.default_epoch_cycles, ...
    'VariableNames', {'denoising_strength', 'epoch_size_in_cycles'});

results = bayesopt(@objective_func, vars, ...
    'MaxObjectiveEvaluations', max_evals, ...
    'InitialX', initial_point, ...
    'PlotFcn', [], ...
    'UseParallel', false, ...
    'Verbose', opts.verbose);

%% 6. Compile Final Outputs & Render Final Artifact Visualization
if isfield(results, 'XAtMinObjective') && ~isempty(results.XAtMinObjective)
    d_opt = results.XAtMinObjective.denoising_strength;
    c_opt = round(results.XAtMinObjective.epoch_size_in_cycles);
elseif isfield(state.best_params, 'denoising_strength')
    d_opt = state.best_params.denoising_strength;
    c_opt = round(state.best_params.epoch_size_in_cycles);
else
    d_opt = 3;
    c_opt = 12;
end

w_opt = opts.smoothing_window_seconds;

% Perform final run using best parameters with artifact visualization enabled
[best_EEGclean, ~, score_opt] = GEDAI(EEGin, d_opt, c_opt, opts.lowcut_frequency, ...
    opts.ref_matrix_type, opts.parallel, opts.visualize, inf, inf, signal_type, w_opt, struct('silent', true));

best_params = struct(...
    'denoising_strength', d_opt, ...
    'epoch_size_in_cycles', c_opt, ...
    'SENSAI_score', score_opt);

end

%% =========================================================================
%% HELPER FUNCTIONS FOR REALTIME VISUALIZATION
%% =========================================================================

function fig = create_realtime_figure(max_evals)
    fig = figure('Name', 'GEDAI Bayesian Optimization', ...
                 'Color', [0.09 0.09 0.13], ...
                 'Position', [100 100 1100 520], ...
                 'NumberTitle', 'off', ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none');
             
    % Panel 1: Convergence
    ax1 = subplot(1, 2, 1);
    set(ax1, 'Color', [0.12 0.12 0.17], ...
             'XColor', [0.7 0.7 0.7], ...
             'YColor', [0.7 0.7 0.7], ...
             'GridColor', [0.25 0.25 0.32], ...
             'GridAlpha', 0.5, ...
             'Box', 'on');
    grid(ax1, 'on');
    xlabel(ax1, 'Evaluation #', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    ylabel(ax1, 'Best SENSAI Score', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    title(ax1, sprintf('Convergence (0 / %d evals  0%%)\nBest SENSAI Score = 0.0', max_evals), ...
          'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
    xlim(ax1, [0 max_evals]);
    ylim(ax1, [0 100]);

    % Panel 2: Current Best Parameters (One bar per hyperparameter)
    ax2 = subplot(1, 2, 2);
    set(ax2, 'Color', [0.12 0.12 0.17], ...
             'XColor', [0.7 0.7 0.7], ...
             'YColor', [0.7 0.7 0.7], ...
             'GridColor', [0.25 0.25 0.32], ...
             'GridAlpha', 0.5, ...
             'Box', 'on');
    grid(ax2, 'on');
    xlabel(ax2, 'Hyperparameter', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    ylabel(ax2, 'Parameter Value', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    title(ax2, sprintf('Current Best Parameters\nSENSAI Score = 0.0'), ...
          'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
    set(ax2, 'XTick', 1:2, 'XTickLabel', {'Denoising Strength', 'Epoch Cycles'});
    ylim(ax2, [0 22]);
    
    drawnow;
end

function update_realtime_figure(state)
    fig = state.fig;
    if ~ishandle(fig), return; end
    
    eval_idx = state.eval_count;
    max_evals = state.max_evals;
    scores = state.scores_history;
    running_best = state.running_best_history;
    best_score = state.best_score;
    
    pct = (eval_idx / max_evals) * 100;
    
    % Panel 1: Convergence
    ax1 = subplot(1, 2, 1, 'Parent', fig);
    cla(ax1);
    hold(ax1, 'on');
    grid(ax1, 'on');
    set(ax1, 'Color', [0.12 0.12 0.17], 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
             'GridColor', [0.25 0.25 0.32], 'GridAlpha', 0.5, 'Box', 'on');
         
    eval_nums = 1:eval_idx;
    
    % Scatter points for individual evaluations (gold/amber)
    scatter(ax1, eval_nums, scores, 35, [0.85 0.65 0.25], 'filled', 'MarkerEdgeColor', 'none');
    
    % Running best score line (bright green)
    plot(ax1, eval_nums, running_best, '-', 'Color', [0.35 0.85 0.55], 'LineWidth', 2.5);
    
    % Highlight best point
    if state.best_eval_idx <= eval_idx
        scatter(ax1, state.best_eval_idx, best_score, 45, [0.35 0.85 0.55], 'filled', 'MarkerEdgeColor', 'w');
    end
    
    xlabel(ax1, 'Evaluation #', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    ylabel(ax1, 'Best SENSAI Score', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    title(ax1, sprintf('Convergence  (%d / %d evals  %.0f%%)\nBest SENSAI Score = %.1f', eval_idx, max_evals, pct, best_score), ...
          'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
    xlim(ax1, [0 max_evals]);
    ylim(ax1, [0 100]);
    hold(ax1, 'off');
    
    % Panel 2: Current Best Parameters (2 Hyperparameters)
    ax2 = subplot(1, 2, 2, 'Parent', fig);
    cla(ax2);
    hold(ax2, 'on');
    grid(ax2, 'on');
    set(ax2, 'Color', [0.12 0.12 0.17], 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
             'GridColor', [0.25 0.25 0.32], 'GridAlpha', 0.5, 'Box', 'on');
         
    if ~isempty(fieldnames(state.best_params))
        d_val = state.best_params.denoising_strength;
        c_val = round(state.best_params.epoch_size_in_cycles);
        
        bar_heights = [d_val, c_val];
        
        hb = bar(ax2, 1:2, bar_heights, 0.4, 'FaceColor', 'flat');
        
        colors = [
            0.22  0.45  0.75;   % Denoising Strength (Blue)
            0.55  0.30  0.65    % Epoch Cycles (Purple)
        ];
        for b_idx = 1:2
            hb.CData(b_idx, :) = colors(b_idx, :);
        end
        
        offset = max(0.8, max(bar_heights) * 0.04);
        text(ax2, 1, d_val + offset, sprintf('%.1f', d_val), ...
             'HorizontalAlignment', 'center', 'Color', [0.95 0.95 0.95], 'FontWeight', 'bold', 'FontSize', 10);
        text(ax2, 2, c_val + offset, sprintf('%d', c_val), ...
             'HorizontalAlignment', 'center', 'Color', [0.95 0.95 0.95], 'FontWeight', 'bold', 'FontSize', 10);
        
        set(ax2, 'XTick', 1:2, 'XTickLabel', {'Denoising Strength', 'Epoch Cycles'});
        max_h = max(bar_heights);
        ylim(ax2, [0 max(22, max_h * 1.2)]);
    else
        ylim(ax2, [0 22]);
        set(ax2, 'XTick', 1:2, 'XTickLabel', {'Denoising Strength', 'Epoch Cycles'});
    end
    
    xlabel(ax2, 'Hyperparameter', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    ylabel(ax2, 'Parameter Value', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    title(ax2, sprintf('Current Best Parameters\nSENSAI Score = %.1f', best_score), ...
          'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
    hold(ax2, 'off');
    
    drawnow limitrate;
end
