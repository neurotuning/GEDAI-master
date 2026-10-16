% [Generalized Eigenvalue De-Artifacting Instrument (GEDAI) v 1.8]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland

function vers = eegplugin_GEDAI(fig, try_strings, catch_strings)
    vers = 'GEDAI v1.8';

    g = fileparts(which('eegplugin_GEDAI'));
    addpath(g);
    addpath(fullfile(g, 'auxiliaries'));

    if nargin < 2 || isempty(try_strings)
        cb_str = 'EEG = pop_GEDAI(EEG); eeglab redraw;';
    else
        cb_str = [try_strings.no_check '[EEG, LASTCOM] = pop_GEDAI(EEG);' catch_strings.new_and_hist];
    end

    % 1. Add under standard EEGLAB Tools menu
    toolsmenu = findobj(fig, 'tag', 'tools');
    if ~isempty(toolsmenu)
        uimenu(toolsmenu, 'label', 'GEDAI Denoising', 'separator', 'on', 'callback', cb_str);
    end

    % 2. Also add as top-level menu bar item
    uimenu(fig, 'label', 'GEDAI', 'callback', cb_str);
end
