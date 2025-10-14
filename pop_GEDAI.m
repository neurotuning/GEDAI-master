%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pop_GEDAI plugin
% This EEGlab GUI is used to call the GEDAI denoising function, and select parameters
%
% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% GNU Affero General Public License v3
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License version 3 
% as published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Affero General Public License for more details.
%
% You should have received a copy of the GNU Affero General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [EEG, com] = pop_GEDAI(EEG, varargin)
%% Default parameter values
artifact_threshold = 'auto';
epoch_size_in_seconds = 1;

% Create an inputParser to handle varargin
p = inputParser;
p.addParameter('artifact_threshold', artifact_threshold, @isnumeric);
p.addParameter('epoch_size', epoch_size_in_seconds, @isnumeric);
p.addParameter('parallel_processing', false, @(parallel_processing) islogical(parallel_processing)); % Add output visual parameter
p.addParameter('visualization_A', false, @(visualization_A) islogical(visualization_A)); % Add output visual parameter
p.parse(varargin{:}); % Parse the input arguments

% Extract parameters from the inputParser results
artifact_threshold = p.Results.artifact_threshold;
epoch_size_in_seconds = p.Results.epoch_size;

% Create GUI for parameter input (rest of the code remains the same)
uilist = { ...    
    {'style' 'text' 'string' 'Denoising strength'}    {'style' 'popupmenu' 'string' '                    auto|                    auto+|                    auto-'} ...
    { 'style' 'text' 'string' 'Epoch size (seconds)' }, ...
    { 'style' 'edit' 'string' num2str(epoch_size_in_seconds) 'tag' 'epoch_size' } ...
    {'style' 'text' 'string' 'Leadfield matrix'}    {'style' 'popupmenu' 'string' '          precomputed|          interpolated'} ...
    { 'style' 'text' 'string' 'Parallel processing' }, ...
    {'style' 'checkbox' 'string' ' requires more RAM' 'tag' 'parallel_processing' 'Value' 1}, ...
    { 'style' 'text' 'string' 'Artifact visualization' }, ...
    {'style' 'checkbox' 'string' ' vis_artifacts from ASR' 'tag' 'visualization_A' 'Value' 1}, ...
};
geometry = { [1, 1] [1, 1] [1, 1] [1, 1] [1, 1]};
title = '  GEDAI denoising |  v1.1  ';

% Get user input
[userInput, ~, ~, out] = inputgui( geometry, uilist, 'help(''GEDAI'')', title);
if isempty(out), return; end

% Extract parameters
epoch_size_in_seconds = str2double(out.epoch_size);

threshold_cell = {'auto','auto+', 'auto-'};
artifact_threshold = threshold_cell{userInput{1,1}};

ref_matrix_cell = {'precomputed','interpolated'};
ref_matrix_type = ref_matrix_cell{userInput{1,3}};

use_parallel = logical(out.parallel_processing);
visualize_artifacts = logical(out.visualization_A);


        [EEG, ~, ~,~, ~, com] = GEDAI(EEG,artifact_threshold, epoch_size_in_seconds,ref_matrix_type,use_parallel,visualize_artifacts);
  
        EEG = eegh(com, EEG); % update EEG.history
    

end
