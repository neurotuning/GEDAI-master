
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
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function vers=eegplugin_GEDAI(fig, try_strings, catch_strings)
% version
vers = 'GEDAI v1.7 - May 2026';

g = fileparts(which('eegplugin_GEDAI'));
addpath(fullfile(g, 'auxiliaries'));

    % Delete any pre-existing GEDAI menu items to avoid duplicates
    old_menu = findobj(fig, 'type', 'uimenu', 'label', 'GEDAI');
    if ~isempty(old_menu)
        delete(old_menu);
    end

    % Add single direct menu item to EEGLAB top menu bar
    uimenu(fig, 'label', 'GEDAI', 'callback', 'EEG = pop_GEDAI(EEG); eeglab redraw');


end