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
artifact_threshold = 'auto';
epoch_size_in_cycles = 12;
lowcut_frequency = 0.5;
ENOVA_threshold = 0.9;

% Create an inputParser to handle varargin
p = inputParser;
addParameter(p, 'artifact_threshold', artifact_threshold);
addParameter(p, 'parallel_processing', false); % Add output visual parameter
addParameter(p, 'visualization_A', false); % Add output visual parameter
p.parse(varargin{:}); % Parse the input arguments

% Create GUI for parameter input (rest of the code remains the same)
uilist = { ...    
    {'style' 'text' 'string' 'Denoising strength'}    {'style' 'popupmenu' 'string' '                    auto|                    auto+|                    auto-|                 bayesopt'} ...
    {'style' 'text' 'string' 'Leadfield matrix'}    {'style' 'popupmenu' 'string' '          precomputed|          interpolated'} ...
    {'style' 'text' 'string' 'Epoch size (wave cycles)'} {'style' 'edit' 'string' num2str(epoch_size_in_cycles) 'tag' 'epoch_size_in_cycles'} ...
    {'style' 'text' 'string' 'Low-cut frequency (Hz)'} {'style' 'edit' 'string' num2str(lowcut_frequency) 'tag' 'lowcut_frequency'} ...
    {} ...
    {'style' 'text' 'string' 'Reject bad epochs:'} {'style' 'checkbox' 'string' '' 'tag' 'reject_by_enova' 'value' 0}, ...
    {'style' 'text' 'string' 'ENOVA Threshold (0-1)'} {'style' 'edit' 'string' num2str(ENOVA_threshold) 'tag' 'ENOVA_threshold'}, ...
    {} ...
    {'style' 'text' 'string' 'Parallel processing ( > RAM):'} {'style' 'checkbox' 'string' '' 'tag' 'parallel_processing' 'Value' 1}, ...
    {'style' 'text' 'string' 'Artifact visualization (from ASR):'} {'style' 'checkbox' 'string' '' 'tag' 'visualization_A' 'Value' 1}, ...
    {'style' 'text' 'string' 'SENSAI visualization:'} {'style' 'checkbox' 'string' '' 'tag' 'visualize_manifold' 'Value' 1}, ...
};
geometry = { [1, 1] [1, 1] [1, 1] [1, 1] [1] [1, 1] [1, 1] [1] [1, 1] [1, 1] [1, 1] };
title = '  GEDAI denoising |  v1.6  ';

% Get user input
[userInput, ~, ~, out] = inputgui( geometry, uilist, 'help(''GEDAI'')', title);
if isempty(out), return; end

threshold_cell = {'auto', 'auto+', 'auto-', 'bayesopt'};
artifact_threshold = threshold_cell{userInput{1}};

ref_matrix_cell = {'precomputed', 'interpolated'};
ref_matrix_type = ref_matrix_cell{userInput{2}};
epoch_size_in_cycles = str2double(out.epoch_size_in_cycles);
lowcut_frequency = str2double(out.lowcut_frequency);

if out.reject_by_enova
    ENOVA_threshold = str2double(out.ENOVA_threshold);
else
    ENOVA_threshold = [];
end

use_parallel = logical(out.parallel_processing);
visualize_artifacts = logical(out.visualization_A);
visualize_manifold = logical(out.visualize_manifold);

[EEG, ~, ~, ~, ~, ~, ~, com] = GEDAI(EEG,artifact_threshold,epoch_size_in_cycles, lowcut_frequency,ref_matrix_type,use_parallel,visualize_artifacts, ENOVA_threshold, [], visualize_manifold);
  
EEG = eegh(com, EEG); % update EEG.history
    

end