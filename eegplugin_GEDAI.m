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


function vers=eegplugin_GEDAI(fig, try_strings, catch_strings)
% version
vers = 'GEDAI v1.0';

g = fileparts(which('eegplugin_GEDAI'));
addpath(fullfile(g, 'auxiliaries'));

    % Add menu item to EEGLAB interface
    menu_item = uimenu(fig, 'label', 'GEDAI', 'callback', 'EEG = pop_GEDAI(EEG);eeglab redraw');


end
