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
%% Default parameter values
epoch_size_in_cycles = 12;
lowcut_frequency = 0.5;
ENOVA_threshold_per_epoch = 0.95;
ENOVA_threshold_per_channel = 0.95;
smoothing_window_seconds_default = Inf;

% Build popup menu entries for output reference selection.
channel_labels = {EEG.chanlocs.labels};
channel_labels_sanitized = strrep(channel_labels, '|', '/');
channel_labels_lower = lower(channel_labels);

% Calibrated proportional center-padding for popupmenu controls in Windows MATLAB
pad_center = @(s) [repmat(' ', 1, max(0, round(36 - ( ...
    0.5 * sum(ismember(strtrim(s), 'ilftjrI1() .-+:;,!')) + ...
    1.2 * sum(ismember(strtrim(s), 'mwMWGOCDQ')) + ...
    0.9 * sum(~ismember(strtrim(s), 'ilftjrI1() .-+:;,!mwMWGOCDQ')))))), strtrim(s)];

output_ref_display_options = {pad_center('AvgRef'), pad_center('REST')};
output_ref_value_options = {'AvgRef', 'REST'};

% Add pairwise average options only when both channels are present.
if any(strcmp(channel_labels_lower, 'm1')) && any(strcmp(channel_labels_lower, 'm2'))
    output_ref_display_options{end+1} = pad_center('M1+M2 (avg)');
    output_ref_value_options{end+1} = '__GEDAI_REF_M1M2_AVG__';
end
if any(strcmp(channel_labels_lower, 'a1')) && any(strcmp(channel_labels_lower, 'a2'))
    output_ref_display_options{end+1} = pad_center('A1+A2 (avg)');
    output_ref_value_options{end+1} = '__GEDAI_REF_A1A2_AVG__';
end
if any(strcmp(channel_labels_lower, 'tp9')) && any(strcmp(channel_labels_lower, 'tp10'))
    output_ref_display_options{end+1} = pad_center('TP9+TP10 (avg)');
    output_ref_value_options{end+1} = '__GEDAI_REF_TP9TP10_AVG__';
end

% If EEG.ref specifies a known recording reference channel omitted from chanlocs, offer it in popup
if isfield(EEG, 'ref') && (ischar(EEG.ref) || (isstring(EEG.ref) && isscalar(EEG.ref)))
    rec_ref_str = strtrim(char(EEG.ref));
    if ~isempty(rec_ref_str) && ~ismember(lower(rec_ref_str), {'average', 'avgref', 'common', 'rest', 'none', 'raw'})
        if ~any(strcmpi(channel_labels, rec_ref_str))
            output_ref_display_options{end+1} = pad_center([rec_ref_str ' (recording ref - leadfield only)']);
            output_ref_value_options{end+1} = rec_ref_str;
        end
    end
end

% Add all individual channels.
for chIdx = 1:numel(channel_labels_sanitized)
    output_ref_display_options{end+1} = pad_center(channel_labels_sanitized{chIdx});
    output_ref_value_options{end+1} = channel_labels{chIdx};
end

ref_channel_popup_options = strjoin(output_ref_display_options, '|');

% Create an inputParser to handle varargin
p = inputParser;
addParameter(p, 'parallel_processing', false); % Add output visual parameter
addParameter(p, 'visualization_A', false); % Add output visual parameter
p.parse(varargin{:}); % Parse the input arguments

% Create GUI for parameter input
uilist = { ...
    {'style' 'text' 'string' 'Epoch size (wave cycles)'} {'style' 'edit' 'string' num2str(epoch_size_in_cycles) 'tag' 'epoch_size_in_cycles' 'horizontalalignment' 'center'} ...
    {'style' 'text' 'string' 'Low-cut frequency (Hz)'} {'style' 'edit' 'string' num2str(lowcut_frequency) 'tag' 'lowcut_frequency' 'horizontalalignment' 'center'} ...
    {'style' 'text' 'string' 'Sliding window (seconds, Inf=all)'} {'style' 'edit' 'string' num2str(smoothing_window_seconds_default) 'tag' 'smoothing_window_seconds' 'horizontalalignment' 'center'} ...
    {'style' 'checkbox' 'string' ' Reject bad channels (ENOVA threshold):' 'tag' 'reject_channels_by_enova' 'value' 0} {'style' 'edit' 'string' num2str(ENOVA_threshold_per_channel) 'tag' 'ENOVA_threshold_per_channel' 'horizontalalignment' 'center'} ...
    {'style' 'checkbox' 'string' ' Reject bad epochs (ENOVA threshold):' 'tag' 'reject_epochs_by_enova' 'value' 0} {'style' 'edit' 'string' num2str(ENOVA_threshold_per_epoch) 'tag' 'ENOVA_threshold_per_epoch' 'horizontalalignment' 'center'} ...
    {} ...
    {'style' 'text' 'string' 'Output reference'} {'style' 'popupmenu' 'string' ref_channel_popup_options 'tag' 'output_reference_popup' 'horizontalalignment' 'center'} ...
    {'style' 'text' 'string' 'Recording reference channel (e.g. FCz; overrides above)'} {'style' 'edit' 'string' '' 'tag' 'custom_reference_channel' 'horizontalalignment' 'center'} ...
    {'style' 'checkbox' 'string' ' Parallel processing ( > RAM)' 'tag' 'parallel_processing' 'Value' 1} ...
    {'style' 'checkbox' 'string' ' Artifact visualization' 'tag' 'visualization_A' 'Value' 1} ...
};
geometry = { [1.4, 1] [1.4, 1] [1.4, 1] [1.4, 1] [1.4, 1] [1] [1.4, 1] [1.4, 1] [1] [1] };
title = '  GEDAI denoising toolbox |  v1.9  ';

% Get user input
[~, ~, ~, out] = inputgui( 'geometry', geometry, 'uilist', uilist, 'helpcom', 'help(''GEDAI'')', 'title', title, 'skipline', 'off');
if isempty(out), return; end

ref_matrix_type = 'auto';

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

% Custom reference write-in takes precedence over popup selection if specified
custom_ref_str = '';
if isfield(out, 'custom_reference_channel') && ~isempty(out.custom_reference_channel)
    if iscell(out.custom_reference_channel)
        custom_ref_str = strtrim(out.custom_reference_channel{1});
    else
        custom_ref_str = strtrim(char(out.custom_reference_channel));
    end
end

if ~isempty(custom_ref_str)
    output_reference_channel = custom_ref_str;
else
    output_reference_channel = strtrim(selected_output_reference_channel);
end

[EEG, ~, ~, ~, ~, ~, ~, com] = GEDAI(EEG, epoch_size_in_cycles, lowcut_frequency, ...
    ref_matrix_type, use_parallel, visualize_artifacts, ENOVA_threshold_per_epoch, ...
    ENOVA_threshold_per_channel, [], smoothing_window_seconds, output_reference_channel);
  
EEG = eegh(com, EEG); % update EEG.history

end
