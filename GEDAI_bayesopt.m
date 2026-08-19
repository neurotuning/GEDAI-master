function [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band, ENOVA_per_channel] = GEDAI_bayesopt(EEGin, varargin)
% GEDAI_BAYESOPT Bayesian Optimization of GEDAI Hyperparameters
%
% Usage:
%   [EEGclean, EEGartifacts, SENSAI_score, ...] = GEDAI_bayesopt(EEGin)
%   [EEGclean, EEGartifacts, SENSAI_score, ...] = GEDAI_bayesopt(EEGin, 'max_evals', 20, ...)
%
% Optimized Hyperparameters:
%   1. Denoising strength (1 to 9)
%   2. Epoch size in wave cycles (2 to 20)
%
% Inputs:
%   EEGin                 - Input EEGLAB structure
%   'max_evals'           - Maximum number of objective evaluations (default: 20)
%   'lowcut_frequency'    - Low-cut frequency in Hz (default: 0.5)
%   'ref_matrix_type'     - Reference leadfield matrix type (default: 'precomputed')
%   'signal_type'         - Signal type: 'eeg' or 'meg' (default: 'eeg')
%   'parallel'            - Enable parallel processing (default: true)
%   'visualize'           - Enable realtime dark-themed visualization (default: false)
%   'denoising_range'     - Range for denoising strength [min, max] (default: [1, 9])
%   'epoch_cycles_range'  - Range for epoch size in wave cycles [min, max] (default: [2, 20])
%   'verbose'             - Verbosity level for bayesopt (default: 0)
%
% Outputs (Identical to GEDAI.m):
%   EEGclean              - Cleaned EEGLAB structure using optimal hyperparameters
%   EEGartifacts          - Removed artifact data
%   SENSAI_score          - Overall SENSAI score (%) for optimal data
%   SENSAI_score_per_band - SENSAI score per frequency band (%)
%   ... (matches GEDAI.m output arguments 1-10)
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
addParameter(p, 'parallel', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'visualize', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'denoising_range', [1, 9], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'epoch_cycles_range', [2, 20], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'gamma_range', [1e-8, 1e-1], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'alpha_range', [0, 1], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'smoothing_window_seconds', Inf, @isnumeric);
addParameter(p, 'blending_method', 'gromov-wasserstein', @(x) ischar(x) || isstring(x));
addParameter(p, 'default_denoising', 7, @isnumeric);
addParameter(p, 'default_epoch_cycles', 12, @isnumeric);
addParameter(p, 'num_comp_per_epoch', 3, @isnumeric);
addParameter(p, 'ENOVA_threshold_per_channel', inf, @isnumeric);
addParameter(p, 'ENOVA_threshold_per_epoch', inf, @isnumeric);
addParameter(p, 'output_reference_channel', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'verbose', 0, @isnumeric);
addParameter(p, 'resampling', true, @(x) islogical(x) || isnumeric(x));
parse(p, EEGin, varargin{:});
opts = p.Results;
opts.parallel = logical(opts.parallel);
opts.visualize = logical(opts.visualize);

max_evals = opts.max_evals;
num_seed_points = opts.num_seed_points;
signal_type = char(opts.signal_type);
is_subject_adapted = ischar(opts.ref_matrix_type) && strcmp(opts.ref_matrix_type, 'subject_adapted');
use_resampling = logical(opts.resampling) && is_subject_adapted;

%% 2. Define Optimization Variables (1 or 2 Hyperparameters)
if is_subject_adapted
    vars = [
        optimizableVariable('subject_adapted_alpha', opts.alpha_range, 'Type', 'real')
    ];
else
    vars = [
        optimizableVariable('denoising_strength', opts.denoising_range, 'Type', 'real')
        optimizableVariable('epoch_size_in_cycles', opts.epoch_cycles_range, 'Type', 'integer')
    ];
end

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
state.G_base = [];
state.G_full = [];
state.A_clean = [];
state.A_clean_per_band = {};
state.current_sample_indices = [];
state.best_sample_indices = [];

if opts.visualize
    state.fig = create_realtime_figure(max_evals, is_subject_adapted);
end

%% 4. Define Objective Function Callback
    function objective = objective_func(X)
        state.eval_count = state.eval_count + 1;
        eval_idx = state.eval_count;
        
        if is_subject_adapted
            d_strength = opts.default_denoising;
            alpha_val = X.subject_adapted_alpha;
            comp_val = opts.num_comp_per_epoch;
            epoch_cycles = opts.default_epoch_cycles;
            
            resample_col_val = false;
            if use_resampling && eval_idx > 1 && ~isempty(state.A_clean)
                total_pat = size(state.A_clean, 2);
                n_ch = size(state.A_clean, 1);
                num_to_samp = min(n_ch, total_pat);
                if total_pat >= num_to_samp
                    samp_idx = randperm(total_pat, num_to_samp);
                else
                    samp_idx = randi(total_pat, 1, num_to_samp);
                end
                resample_col_val = samp_idx;
                state.current_sample_indices = samp_idx;
            else
                state.current_sample_indices = [];
            end
            
            gedai_extra = struct('subject_adapted_alpha', alpha_val, ...
                                 'regularization_lambda', 0.05, ...
                                 'blending_method', opts.blending_method, ...
                                 'num_comp_per_epoch', comp_val, ...
                                 'G_base', state.G_base, ...
                                 'G_full', state.G_full, ...
                                 'A_clean', state.A_clean, ...
                                 'A_clean_per_band', {state.A_clean_per_band}, ...
                                 'resample_columns', resample_col_val, ...
                                 'silent', true);
        else
            d_strength = X.denoising_strength;
            epoch_cycles = round(X.epoch_size_in_cycles);
            gedai_extra = struct('silent', true);
        end
        win_sec_eval = opts.smoothing_window_seconds;
        
        try
            [EEGclean, EEGartifacts, sensai_val, ~, artifact_thresh_band, mean_enova, ~] = ...
                GEDAI(EEGin, d_strength, epoch_cycles, opts.lowcut_frequency, ...
                      opts.ref_matrix_type, opts.parallel, false, inf, inf, ...
                      signal_type, win_sec_eval, gedai_extra);
            
            % Cache leadfield Gram matrix and A_clean on first successful evaluation for fast reuse
            if is_subject_adapted && isstruct(EEGclean) && isfield(EEGclean, 'etc') && isfield(EEGclean.etc, 'GEDAI')
                if isempty(state.G_base) && isfield(EEGclean.etc.GEDAI, 'G_base'), state.G_base = EEGclean.etc.GEDAI.G_base; end
                if isempty(state.G_full) && isfield(EEGclean.etc.GEDAI, 'G_full'), state.G_full = EEGclean.etc.GEDAI.G_full; end
                if isempty(state.A_clean) && isfield(EEGclean.etc.GEDAI, 'A_clean'), state.A_clean = EEGclean.etc.GEDAI.A_clean; end
                if isempty(state.A_clean_per_band) && isfield(EEGclean.etc.GEDAI, 'A_clean_per_band'), state.A_clean_per_band = EEGclean.etc.GEDAI.A_clean_per_band; end
            end
            
            sensai_score = sensai_val;
            if isnan(sensai_score) || isinf(sensai_score)
                sensai_score = 0;
            end
            eval_score = sensai_score;
        catch ME
            warning('GEDAI evaluation %d failed: %s', eval_idx, ME.message);
            sensai_score = 0;
            eval_score = 0;
            artifact_thresh_band = [];
            EEGclean = [];
        end
        
        objective = -eval_score; % bayesopt minimizes objective
        
        % Update tracking state
        state.scores_history(eval_idx) = eval_score;
        if eval_idx == 1
            current_max = eval_score;
        else
            current_max = max(state.running_best_history(end), eval_score);
        end
        state.running_best_history(eval_idx) = current_max;
        
        % Check if new best score found
        if eval_score > state.best_score
            state.best_score = eval_score;
            state.best_eval_idx = eval_idx;
            state.best_sample_indices = state.current_sample_indices;
            if is_subject_adapted
                state.best_params = struct(...
                    'denoising_strength', d_strength, ...
                    'subject_adapted_alpha', alpha_val, ...
                    'num_comp_per_epoch', comp_val, ...
                    'blending_method', opts.blending_method, ...
                    'resampling', use_resampling, ...
                    'sampled_indices', state.best_sample_indices, ...
                    'SENSAI_score', sensai_score);
            else
                state.best_params = struct(...
                    'denoising_strength', d_strength, ...
                    'epoch_size_in_cycles', epoch_cycles, ...
                    'SENSAI_score', sensai_score);
            end
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

%% 5. Run Bayesian Optimization (Warm-started with alpha=1.0 pure leadfield baseline as Evaluation #1)
if is_subject_adapted
    initial_point = table(1.0, 'VariableNames', {'subject_adapted_alpha'});
else
    initial_point = table(opts.default_denoising, opts.default_epoch_cycles, ...
        'VariableNames', {'denoising_strength', 'epoch_size_in_cycles'});
end

results = bayesopt(@objective_func, vars, ...
    'MaxObjectiveEvaluations', max_evals, ...
    'InitialX', initial_point, ...
    'PlotFcn', [], ...
    'UseParallel', false, ...
    'Verbose', opts.verbose);

%% 6. Compile Final Outputs & Render Final Artifact Visualization
if is_subject_adapted
    if isfield(results, 'XAtMinObjective') && ~isempty(results.XAtMinObjective)
        a_opt = results.XAtMinObjective.subject_adapted_alpha;
    elseif isfield(state.best_params, 'subject_adapted_alpha')
        a_opt = state.best_params.subject_adapted_alpha;
    else
        a_opt = 1.0;
    end
    comp_opt = opts.num_comp_per_epoch;
    d_opt = opts.default_denoising;
    c_opt = opts.default_epoch_cycles;

    final_gedai_extra = struct('subject_adapted_alpha', a_opt, ...
                               'regularization_lambda', 0.05, ...
                               'blending_method', opts.blending_method, ...
                               'num_comp_per_epoch', comp_opt, ...
                               'silent', true);
    if use_resampling && ~isempty(state.best_sample_indices) && ~isempty(state.A_clean)
        final_gedai_extra.A_clean = state.A_clean;
        final_gedai_extra.resample_columns = state.best_sample_indices;
    end

    [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band, ENOVA_per_channel] = ...
        GEDAI(EEGin, d_opt, c_opt, opts.lowcut_frequency, ...
        opts.ref_matrix_type, opts.parallel, opts.visualize, opts.ENOVA_threshold_per_epoch, opts.ENOVA_threshold_per_channel, signal_type, opts.smoothing_window_seconds, opts.output_reference_channel, ...
        final_gedai_extra);

    best_params = struct(...
        'denoising_strength', d_opt, ...
        'subject_adapted_alpha', a_opt, ...
        'num_comp_per_epoch', comp_opt, ...
        'blending_method', opts.blending_method, ...
        'resampling', use_resampling, ...
        'sampled_indices', state.best_sample_indices, ...
        'SENSAI_score', SENSAI_score);
else
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

    % Perform final run using best parameters with artifact visualization enabled
    [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band, ENOVA_per_channel] = ...
        GEDAI(EEGin, d_opt, c_opt, opts.lowcut_frequency, ...
        opts.ref_matrix_type, opts.parallel, opts.visualize, opts.ENOVA_threshold_per_epoch, opts.ENOVA_threshold_per_channel, signal_type, opts.smoothing_window_seconds, opts.output_reference_channel, struct('silent', true));

    best_params = struct(...
        'denoising_strength', d_opt, ...
        'epoch_size_in_cycles', c_opt, ...
        'SENSAI_score', SENSAI_score);
end

if isstruct(EEGclean) && isfield(EEGclean, 'etc')
    EEGclean.etc.GEDAI.bayesopt_best_params = best_params;
    EEGclean.etc.GEDAI.bayesopt_results = results;
end

%% Print Final Optimal Settings Summary to Command Window
disp([newline '==================================================']);
disp('   GEDAI BAYESIAN OPTIMIZATION: OPTIMAL SETTINGS');
disp('==================================================');
fprintf('  Leadfield Model:       %s\n', opts.ref_matrix_type);
fprintf('  Denoising Strength:    %.2f\n', d_opt);
if is_subject_adapted
    fprintf('  Blending Method:       %s\n', opts.blending_method);
    fprintf('  Subject-Adapted Alpha: %.3f (%.1f%% empirical, %.1f%% leadfield)\n', ...
        a_opt, (1 - a_opt) * 100, a_opt * 100);
    fprintf('  Components per Epoch:  %d components\n', comp_opt);
    fprintf('  Regularization Lambda: 0.05 (5%% isotropic shrinkage)\n');
    if use_resampling
        if ~isempty(state.best_sample_indices)
            fprintf('  Subspace Resampling:   Enabled (%d columns sampled from %d patterns)\n', ...
                length(state.best_sample_indices), size(state.A_clean, 2));
        else
            fprintf('  Subspace Resampling:   Enabled (baseline full set optimal)\n');
        end
    else
        fprintf('  Subspace Resampling:   Disabled (all %d empirical patterns used)\n', ...
            size(state.A_clean, 2));
    end
    fprintf('  Epoch Size in Cycles:  %d cycles\n', c_opt);
else
    fprintf('  Epoch Size in Cycles:  %d cycles\n', c_opt);
end
fprintf('  Low-Cut Frequency:     %.2f Hz\n', opts.lowcut_frequency);
if opts.smoothing_window_seconds ~= Inf
    fprintf('  Sliding Window:        %.1f s\n', opts.smoothing_window_seconds);
else
    fprintf('  Sliding Window:        Disabled (Global)\n');
end
fprintf('  ----------------------------------------------\n');
if opts.ENOVA_threshold_per_channel < inf
    fprintf('  Bad Channel Threshold: %.0f%% ENOVA\n', opts.ENOVA_threshold_per_channel * 100);
end
if opts.ENOVA_threshold_per_epoch < inf
    fprintf('  Bad Epoch Threshold:   %.0f%% ENOVA\n', opts.ENOVA_threshold_per_epoch * 100);
end
if ~isempty(opts.output_reference_channel)
    fprintf('  Output Reference:      %s\n', opts.output_reference_channel);
end
fprintf('  Optimal SENSAI Score:  %.2f %%\n', SENSAI_score);
fprintf('  Mean ENOVA:            %.2f %%\n', mean_ENOVA * 100);
fprintf('  Total Evaluations:     %d\n', state.eval_count);
disp(['==================================================' newline]);

end

%% =========================================================================
%% HELPER FUNCTIONS FOR REALTIME VISUALIZATION
%% =========================================================================

function fig = create_realtime_figure(max_evals, is_subject_adapted)
    if nargin < 2, is_subject_adapted = false; end
    fig = figure('Name', 'GEDAI Bayesian Optimization (SENSAI)', ...
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
    title(ax2, 'Current Best Parameters\nSENSAI Score = 0.0', ...
          'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
    if is_subject_adapted
        set(ax2, 'XTick', 1, 'XTickLabel', {'Subject Alpha'});
        ylim(ax2, [0 1.25]);
    else
        set(ax2, 'XTick', 1:2, 'XTickLabel', {'Denoising Strength', 'Epoch Cycles'});
        ylim(ax2, [0 22]);
    end
    
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
    
    % Panel 2: Current Best Parameters
    ax2 = subplot(1, 2, 2, 'Parent', fig);
    cla(ax2);
    hold(ax2, 'on');
    grid(ax2, 'on');
    set(ax2, 'Color', [0.12 0.12 0.17], 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
             'GridColor', [0.25 0.25 0.32], 'GridAlpha', 0.5, 'Box', 'on');
         
    if ~isempty(fieldnames(state.best_params))
        if isfield(state.best_params, 'subject_adapted_alpha')
            a_val = state.best_params.subject_adapted_alpha;
            
            bar_heights = a_val;
            hb = bar(ax2, 1, bar_heights, 0.4, 'FaceColor', 'flat');
            hb.CData(1, :) = [0.22  0.45  0.75];   % Alpha (Blue)
            
            lbl1 = sprintf('Alpha = %.3f\n(%.0f%% Emp, %.0f%% Lead)', a_val, (1 - a_val) * 100, a_val * 100);
            
            text(ax2, 1, bar_heights(1) + 0.05, lbl1, ...
                 'HorizontalAlignment', 'center', 'Color', [0.95 0.95 0.95], 'FontWeight', 'bold', 'FontSize', 10);
            
            set(ax2, 'XTick', 1, 'XTickLabel', {'Subject Alpha'});
            ylim(ax2, [0, 1.25]);
        else
            d_val = state.best_params.denoising_strength;
            c_val = round(state.best_params.epoch_size_in_cycles);
            bar_heights = [d_val, c_val];
            lbl1 = sprintf('%.1f', d_val);
            lbl2 = sprintf('%d', c_val);
            xticklabels_str = {'Denoising Strength', 'Epoch Cycles'};
            ylim_max = max(22, max(bar_heights) * 1.2);
            ylim_min = 0;
            
            hb = bar(ax2, 1:2, bar_heights, 0.4, 'FaceColor', 'flat');
            colors = [
                0.22  0.45  0.75;   % Parameter 1 (Blue)
                0.55  0.30  0.65    % Parameter 2 (Purple)
            ];
            for b_idx = 1:2
                hb.CData(b_idx, :) = colors(b_idx, :);
            end
            
            offset = max(0.4, abs(max(bar_heights)) * 0.04);
            text(ax2, 1, bar_heights(1) + offset, lbl1, ...
                 'HorizontalAlignment', 'center', 'Color', [0.95 0.95 0.95], 'FontWeight', 'bold', 'FontSize', 10);
            text(ax2, 2, bar_heights(2) + offset, lbl2, ...
                 'HorizontalAlignment', 'center', 'Color', [0.95 0.95 0.95], 'FontWeight', 'bold', 'FontSize', 10);
            
            set(ax2, 'XTick', 1:2, 'XTickLabel', xticklabels_str);
            ylim(ax2, [ylim_min, ylim_max]);
        end
    else
        ylim(ax2, [0 22]);
        set(ax2, 'XTick', 1, 'XTickLabel', {'Subject Alpha'});
    end
    
    xlabel(ax2, 'Hyperparameter', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    ylabel(ax2, 'Parameter Value', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]);
    title(ax2, sprintf('Current Best Parameters\nSENSAI Score = %.1f', best_score), ...
          'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
    hold(ax2, 'off');
    
    drawnow limitrate;
end
