%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pop_GEDAI plugin
% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% Copyright (C) [2025] Tomas Ros & Abele Michela - dr.t.ros@gmail.com

function [EEG, com] = pop_GEDAI(EEG, varargin)

artifact_threshold            = 'auto';
epoch_size_in_cycles          = 12;
lowcut_frequency              = 0.5;
ENOVA_threshold               = 0.9;
smoothing_window_minutes_default = Inf;
bayesopt_max_evals_default    = 15;

p = inputParser;
addParameter(p, 'artifact_threshold',  artifact_threshold);
addParameter(p, 'parallel_processing', false);
addParameter(p, 'visualization_A',     false);
p.parse(varargin{:});

uilist = { ...
    {'style' 'text' 'string' 'Denoising strength'} {'style' 'popupmenu' 'string' '                    auto|                    auto+|                    auto-'} ...
    {'style' 'text' 'string' 'Leadfield matrix'}   {'style' 'popupmenu' 'string' '          precomputed|          interpolated'} ...
    {'style' 'text' 'string' 'Epoch size (wave cycles)'} {'style' 'edit' 'string' num2str(epoch_size_in_cycles) 'tag' 'epoch_size_in_cycles'} ...
    {'style' 'text' 'string' 'Low-cut frequency (Hz)'} {'style' 'edit' 'string' num2str(lowcut_frequency) 'tag' 'lowcut_frequency'} ...
    {'style' 'text' 'string' 'Sliding window (minutes, Inf=whole file)'} {'style' 'edit' 'string' num2str(smoothing_window_minutes_default) 'tag' 'smoothing_window_minutes'} ...
    {'style' 'text' 'string' 'BayesOpt window:'} {'style' 'checkbox' 'string' '' 'tag' 'use_bayesopt' 'value' 0} {'style' 'text' 'string' 'Max evaluations:'} {'style' 'edit' 'string' num2str(bayesopt_max_evals_default) 'tag' 'bayesopt_max_evals'} ...
    {} ...
    {'style' 'text' 'string' 'Reject bad epochs:'} {'style' 'checkbox' 'string' '' 'tag' 'reject_by_enova' 'value' 0} ...
    {'style' 'text' 'string' 'ENOVA Threshold (0-1)'} {'style' 'edit' 'string' num2str(ENOVA_threshold) 'tag' 'ENOVA_threshold'} ...
    {} ...
    {'style' 'text' 'string' 'Parallel processing ( > RAM):'} {'style' 'checkbox' 'string' '' 'tag' 'parallel_processing' 'value' 1} ...
    {'style' 'text' 'string' 'Artifact visualization (from ASR):'} {'style' 'checkbox' 'string' '' 'tag' 'visualization_A' 'value' 1} ...
    {'style' 'text' 'string' 'SENSAI visualization:'} {'style' 'checkbox' 'string' '' 'tag' 'visualize_manifold' 'value' 1} ...
};
geometry = { [1,1] [1,1] [1,1] [1,1] [1,1] [1,1,1,1] [1] [1,1] [1,1] [1] [1,1] [1,1] [1,1] };
title = '  GEDAI denoising |  v1.6  ';

[userInput, ~, ~, out] = inputgui(geometry, uilist, 'help(''GEDAI'')', title);
if isempty(out), return; end

threshold_cell     = {'auto', 'auto+', 'auto-'};
artifact_threshold = threshold_cell{userInput{1}};

ref_matrix_cell = {'precomputed', 'interpolated'};
ref_matrix_type = ref_matrix_cell{userInput{2}};

epoch_size_in_cycles = str2double(out.epoch_size_in_cycles);
lowcut_frequency     = str2double(out.lowcut_frequency);

smoothing_window_minutes = str2double(out.smoothing_window_minutes);
if isnan(smoothing_window_minutes)
    smoothing_window_minutes = Inf;
end

use_bayesopt       = logical(out.use_bayesopt);
bayesopt_max_evals = round(str2double(out.bayesopt_max_evals));
if isnan(bayesopt_max_evals) || bayesopt_max_evals < 1
    bayesopt_max_evals = bayesopt_max_evals_default;
end

if out.reject_by_enova
    ENOVA_threshold = str2double(out.ENOVA_threshold);
else
    ENOVA_threshold = [];
end

use_parallel        = logical(out.parallel_processing);
visualize_artifacts = logical(out.visualization_A);
visualize_manifold  = logical(out.visualize_manifold);

if use_bayesopt
    disp('Running Bayesian optimization (smoothing window + epoch cycles)...');
    [optimal_params, ~] = optimize_GEDAI_smoothing(EEG, ...
        'artifact_threshold',  artifact_threshold, ...
        'lowcut_frequency',    lowcut_frequency, ...
        'ref_matrix_type',     ref_matrix_type, ...
        'signal_type',         'eeg', ...
        'MaxEvaluations',      bayesopt_max_evals);
    smoothing_window_minutes = optimal_params.smoothing_window_minutes;
    epoch_size_in_cycles     = optimal_params.epoch_size_in_cycles;
    fprintf('BayesOpt complete: window=%.3f min, cycles=%d, SENSAI=%.4f\n', ...
        smoothing_window_minutes, epoch_size_in_cycles, optimal_params.SENSAI_score);
end

[EEG, ~, ~, ~, ~, ~, ~, com] = GEDAI(EEG, artifact_threshold, epoch_size_in_cycles, ...
    lowcut_frequency, ref_matrix_type, use_parallel, visualize_artifacts, ...
    ENOVA_threshold, [], visualize_manifold, smoothing_window_minutes);

EEG = eegh(com, EEG);

end
