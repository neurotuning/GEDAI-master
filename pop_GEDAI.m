%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pop_GEDAI plugin
% This EEGlab GUI is used to call the GEDAI denoising function, and select parameters
%
% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [EEG, com] = pop_GEDAI(EEG, varargin)
%% 0. Handle Missing/Empty Input Dataset
if nargin < 1 || isempty(EEG)
    if evalin('base', 'exist(''EEG'', ''var'')')
        EEG = evalin('base', 'EEG');
    elseif evalin('base', 'exist(''ALLEEG'', ''var'')')
        ALLEEG_temp = evalin('base', 'ALLEEG');
        if ~isempty(ALLEEG_temp)
            EEG = ALLEEG_temp(1);
        end
    else
        if exist('pop_loadset', 'file')
            EEG = pop_loadset();
        else
            [fname, fpath] = uigetfile('*.set', 'Select EEG dataset');
            if isequal(fname, 0), return; end
            data_st = load(fullfile(fpath, fname), '-mat');
            if isfield(data_st, 'EEG'), EEG = data_st.EEG; else, EEG = data_st; end
        end
    end
end

%% Default parameter values
artifact_threshold = 'auto';
epoch_size_in_cycles = 12;
lowcut_frequency = 0.5;
ENOVA_threshold_per_epoch = 0.95;
ENOVA_threshold_per_channel = 0.95;
smoothing_window_seconds_default = Inf;

% Build popup menu entries for output reference selection.
channel_labels = {EEG.chanlocs.labels};
channel_labels_sanitized = strrep(channel_labels, '|', '/');
channel_labels_lower = lower(channel_labels);

output_ref_display_options = {'AvgRef', 'REST'};
output_ref_value_options = {'AvgRef', 'REST'};

% Add pairwise average options only when both channels are present.
if any(strcmp(channel_labels_lower, 'm1')) && any(strcmp(channel_labels_lower, 'm2'))
    output_ref_display_options{end+1} = 'M1+M2 (avg)';
    output_ref_value_options{end+1} = '__GEDAI_REF_M1M2_AVG__';
end
if any(strcmp(channel_labels_lower, 'a1')) && any(strcmp(channel_labels_lower, 'a2'))
    output_ref_display_options{end+1} = 'A1+A2 (avg)';
    output_ref_value_options{end+1} = '__GEDAI_REF_A1A2_AVG__';
end
if any(strcmp(channel_labels_lower, 'tp9')) && any(strcmp(channel_labels_lower, 'tp10'))
    output_ref_display_options{end+1} = 'TP9+TP10 (avg)';
    output_ref_value_options{end+1} = '__GEDAI_REF_TP9TP10_AVG__';
end

% Add all individual channels.
for chIdx = 1:numel(channel_labels_sanitized)
    output_ref_display_options{end+1} = channel_labels_sanitized{chIdx};
    output_ref_value_options{end+1} = channel_labels{chIdx};
end

ref_channel_popup_options = strjoin(output_ref_display_options, '|');

% Create an inputParser to handle varargin
p = inputParser;
addParameter(p, 'artifact_threshold', artifact_threshold);
addParameter(p, 'parallel_processing', false); % Add output visual parameter
addParameter(p, 'visualization_A', false); % Add output visual parameter
addParameter(p, 'run_bayesopt', true);
addParameter(p, 'bayesopt_max_evals', 20);
p.parse(varargin{:}); % Parse the input arguments

initial_bayesopt_val = 1;
if isfield(p.Results, 'run_bayesopt')
    if islogical(p.Results.run_bayesopt)
        initial_bayesopt_val = double(p.Results.run_bayesopt);
    elseif isnumeric(p.Results.run_bayesopt)
        initial_bayesopt_val = double(p.Results.run_bayesopt > 0);
    end
end

% Create GUI for parameter input (rest of the code remains the same)
uilist = { ...    
    {'style' 'text' 'string' 'Denoising strength'}    {'style' 'popupmenu' 'string' '                    auto|                    auto+|                    auto-'} ...
    {'style' 'text' 'string' 'Leadfield matrix'}    {'style' 'popupmenu' 'string' '          precomputed|          interpolated|          warped'} ...
    {'style' 'text' 'string' 'Epoch size (wave cycles)'} {'style' 'edit' 'string' num2str(epoch_size_in_cycles) 'tag' 'epoch_size_in_cycles'} ...
    {'style' 'text' 'string' 'Low-cut frequency (Hz)'} {'style' 'edit' 'string' num2str(lowcut_frequency) 'tag' 'lowcut_frequency'} ...
    {'style' 'text' 'string' 'Sliding window (in seconds, Inf=whole file)'} {'style' 'edit' 'string' num2str(smoothing_window_seconds_default) 'tag' 'smoothing_window_seconds'} ...
    {} ...
    {'style' 'text' 'string' 'Outer Loop Optimization:'} {'style' 'checkbox' 'string' 'Run Bayesian Optimization (bayesopt)' 'tag' 'run_bayesopt' 'Value' initial_bayesopt_val}, ...
    {'style' 'text' 'string' 'Bayesopt Max Evaluations'} {'style' 'edit' 'string' num2str(p.Results.bayesopt_max_evals) 'tag' 'bayesopt_max_evals'}, ...
    {} ...
    {'style' 'text' 'string' 'Bad Epoch Detection Threshold (ENOVA)'} {'style' 'edit' 'string' num2str(ENOVA_threshold_per_epoch) 'tag' 'ENOVA_threshold_per_epoch'}, ...
    {'style' 'text' 'string' 'Reject bad epochs:'} {'style' 'checkbox' 'string' '' 'tag' 'reject_epochs_by_enova' 'value' 0}, ...
    {} ...
    {'style' 'text' 'string' 'Bad Channel Detection Threshold (ENOVA)'} {'style' 'edit' 'string' num2str(ENOVA_threshold_per_channel) 'tag' 'ENOVA_threshold_per_channel'}, ...
    {'style' 'text' 'string' 'Reject bad channels:'} {'style' 'checkbox' 'string' '' 'tag' 'reject_channels_by_enova' 'value' 0}, ...
    {} ...
    {'style' 'text' 'string' 'Output reference'} {'style' 'popupmenu' 'string' ref_channel_popup_options 'tag' 'output_reference_popup'}, ...
    {} ...
    {'style' 'text' 'string' 'Parallel processing ( > RAM):'} {'style' 'checkbox' 'string' '' 'tag' 'parallel_processing' 'Value' 1}, ...
    {'style' 'text' 'string' 'Artifact visualization:'} {'style' 'checkbox' 'string' '' 'tag' 'visualization_A' 'Value' 1}, ...
};
geometry = { [1, 1] [1, 1] [1, 1] [1, 1] [1, 1] [1] [1.2, 1] [1, 1] [1] [1, 1] [1, 1] [1] [1, 1] [1, 1] [1] [1, 1] [1] [1, 1] [1, 1] };
title = '  GEDAI denoising toolbox |  v1.7  ';

% Get user input
[userInput, ~, ~, out] = inputgui( geometry, uilist, 'help(''GEDAI'')', title);
if isempty(out), return; end

threshold_cell = {'auto', 'auto+', 'auto-'};
artifact_threshold = threshold_cell{userInput{1}};

ref_matrix_cell = {'precomputed', 'interpolated', 'warped'};
ref_matrix_type = ref_matrix_cell{userInput{2}};
epoch_size_in_cycles = str2double(out.epoch_size_in_cycles);
lowcut_frequency = str2double(out.lowcut_frequency);

% Parse smoothing window (allow 'Inf' string)
smoothing_window_seconds = str2double(out.smoothing_window_seconds);
if isnan(smoothing_window_seconds)
    smoothing_window_seconds = Inf;
end

if out.reject_epochs_by_enova
    ENOVA_threshold_per_epoch = str2double(out.ENOVA_threshold_per_epoch);
else
    ENOVA_threshold_per_epoch = [];
end

if out.reject_channels_by_enova
    ENOVA_threshold_per_channel = str2double(out.ENOVA_threshold_per_channel);
else
    ENOVA_threshold_per_channel = [];
end

selected_ref_popup_raw = out.output_reference_popup;
if iscell(selected_ref_popup_raw)
    selected_ref_popup_raw = selected_ref_popup_raw{1};
end

if isnumeric(selected_ref_popup_raw) || islogical(selected_ref_popup_raw)
    selected_ref_popup_index = double(selected_ref_popup_raw(1));
elseif ischar(selected_ref_popup_raw) || (isstring(selected_ref_popup_raw) && isscalar(selected_ref_popup_raw))
    selected_ref_popup_index = str2double(char(selected_ref_popup_raw));
else
    selected_ref_popup_index = 1;
end

if ~isfinite(selected_ref_popup_index) || isempty(selected_ref_popup_index)
    selected_ref_popup_index = 1;
end

selected_ref_popup_index = max(1, round(selected_ref_popup_index));
max_ref_popup_index = numel(output_ref_value_options);
selected_ref_popup_index = min(selected_ref_popup_index, max_ref_popup_index);
selected_output_reference_channel = output_ref_value_options{selected_ref_popup_index};

use_parallel = logical(out.parallel_processing);
visualize_artifacts = logical(out.visualization_A);

% Popup-only behavior: index 1 is AvgRef; any other index applies channel re-reference.
output_reference_channel = strtrim(selected_output_reference_channel);

if out.run_bayesopt
    bayesopt_max_evals = str2double(out.bayesopt_max_evals);
    if isnan(bayesopt_max_evals) || bayesopt_max_evals <= 0
        bayesopt_max_evals = 20;
    end
    
    switch artifact_threshold
        case 'auto', default_d = 7;
        case 'auto+', default_d = 8.5;
        case 'auto-', default_d = 4;
        otherwise
            if isnumeric(artifact_threshold), default_d = artifact_threshold; else, default_d = 7; end
    end
    
    [best_params, EEG, results] = GEDAI_bayesopt(EEG, ...
        'max_evals', bayesopt_max_evals, ...
        'lowcut_frequency', lowcut_frequency, ...
        'ref_matrix_type', ref_matrix_type, ...
        'smoothing_window_seconds', smoothing_window_seconds, ...
        'default_denoising', default_d, ...
        'default_epoch_cycles', epoch_size_in_cycles, ...
        'parallel', use_parallel, ...
        'visualize', visualize_artifacts);
        
    com = sprintf('EEG = GEDAI_bayesopt(EEG, ''max_evals'', %d, ''lowcut_frequency'', %g, ''ref_matrix_type'', ''%s'');', ...
        bayesopt_max_evals, lowcut_frequency, ref_matrix_type);
else
    [EEG, ~, ~, ~, ~, ~, ~, com] = GEDAI(EEG, artifact_threshold, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, use_parallel, visualize_artifacts, ENOVA_threshold_per_epoch, ENOVA_threshold_per_channel, [], smoothing_window_seconds, output_reference_channel);
end
  
EEG = eegh(com, EEG); % update EEG.history
    

end
