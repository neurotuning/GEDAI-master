% build_empirical_residual_networks.m
% Constructs the precomputed 343-electrode empirical residual network covariances
% for the 9 frequency bands in GEDAI, using the independent training set of
% empirical clean EEG datasets from the BEAR benchmark (subjects 1, 3, 5, 7, 9).
%
% Output file: GEDAI_empirical_residual_networks.mat in the auxiliaries/ folder.

clear; clc;
fprintf('========================================================================\n');
fprintf('  BUILDING EMPIRICAL RESIDUAL NETWORK COVARIANCES (343 ELECTRODES)\n');
fprintf('========================================================================\n\n');

% Paths
plugin_dir = fileparts(fileparts(mfilename('fullpath')));
if isempty(plugin_dir) || ~exist(fullfile(plugin_dir, 'GEDAI.m'), 'file')
    plugin_dir = 'c:\Users\Ros\Documents\MATLAB\eeglab2025.0.0\plugins\GEDAI-master';
end
aux_dir = fullfile(plugin_dir, 'auxiliaries');
addpath(plugin_dir);
addpath(aux_dir);
addpath('C:\Users\Ros\Documents\MATLAB\eeglab2025.0.0');
eeglab nogui;

% Load 343 template leadfield
lf_path = fullfile(aux_dir, 'fsavLEADFIELD_4_GEDAI.mat');
if ~exist(lf_path, 'file')
    error('Leadfield template not found at: %s', lf_path);
end
L = load(lf_path, 'leadfield4GEDAI');
tmpl_locs = L.leadfield4GEDAI.EEG.chanlocs;
n_tmpl = length(tmpl_locs);
fprintf('Loaded template with %d electrodes.\n', n_tmpl);

% Locate Clean EEG datasets
clean_dir = 'C:\Users\Ros\Documents\EEG data\BEAR\analysis code\BEAR_EEG_datasets\BEAR-main\EMPIRICAL simulation data\CLEAN EEG';
DirClean = dir(fullfile(clean_dir, '*.set'));
if isempty(DirClean)
    error('Clean EEG datasets not found at: %s', clean_dir);
end

train_idx = [1, 3, 5, 7, 9];
fprintf('Using independent training clean datasets: %s\n', mat2str(train_idx));

band_names = { ...
    'Broadband (0.5-100Hz)', ...
    'Gamma (50-100Hz)', ...
    'Low Gamma (25-50Hz)', ...
    'Beta (12.5-25Hz)', ...
    'Alpha (6.25-12.5Hz)', ...
    'Theta (3.1-6.25Hz)', ...
    'Delta (1.56-3.12Hz)', ...
    'Sub-delta (0.78-1.56Hz)', ...
    'Slow (0.39-0.78Hz)' ...
};
n_bands = length(band_names); % 9
n_wavelet_bands = 8;

optimal_gammas = [0.50, 0.00, 0.00, 0.50, 0.50, 0.00, 0.20, 0.35, 0.10];

C_343_avref = cell(1, n_bands);
C_343_raw   = cell(1, n_bands);
C_27_avref  = cell(1, n_bands);
for b = 1:n_bands
    C_343_avref{b} = zeros(n_tmpl, n_tmpl);
    C_343_raw{b}   = zeros(n_tmpl, n_tmpl);
    C_27_avref{b}  = zeros(27, 27);
end

chanlocs_27 = {};

for t = 1:length(train_idx)
    s_idx = train_idx(t);
    clean_file = fullfile(clean_dir, DirClean(s_idx).name);
    fprintf('Processing subject %d (%s)...\n', s_idx, DirClean(s_idx).name);
    EEG_tr = pop_loadset('filename', DirClean(s_idx).name, 'filepath', clean_dir);
    if isempty(chanlocs_27)
        chanlocs_27 = {EEG_tr.chanlocs.labels};
    end
    
    % Exact 27-channel empirical covariance accumulation
    d_tr_raw = EEG_tr.data;
    d_tr_av  = d_tr_raw - mean(d_tr_raw, 1);
    c_27_bb  = (d_tr_av * d_tr_av') / size(d_tr_av, 2);
    c_27_bb  = real((c_27_bb + c_27_bb') / 2);
    if trace(c_27_bb) > 0, c_27_bb = c_27_bb * (27 / trace(c_27_bb)); end
    C_27_avref{1} = C_27_avref{1} + c_27_bb / length(train_idx);
    
    for wb = 1:n_wavelet_bands
        d_wb_27 = stateful_modwt_single_band(d_tr_av', 'haar', 7, wb)';
        c_wb_27 = (d_wb_27 * d_wb_27') / size(d_wb_27, 2);
        c_wb_27 = real((c_wb_27 + c_wb_27') / 2);
        if trace(c_wb_27) > 0, c_wb_27 = c_wb_27 * (27 / trace(c_wb_27)); end
        C_27_avref{wb + 1} = C_27_avref{wb + 1} + c_wb_27 / length(train_idx);
    end
    
    % Interpolate to 343 OpenMEEG template electrodes
    EEG_343 = eeg_interp(EEG_tr, tmpl_locs, 'spherical');
    d_raw = EEG_343.data;
    d_av  = d_raw - mean(d_raw, 1);
    
    % Band 1: Broadband (0.5 - 100 Hz)
    % Raw
    c_raw_bb = (d_raw * d_raw') / size(d_raw, 2);
    c_raw_bb = real((c_raw_bb + c_raw_bb') / 2);
    if trace(c_raw_bb) > 0, c_raw_bb = c_raw_bb * (n_tmpl / trace(c_raw_bb)); end
    C_343_raw{1} = C_343_raw{1} + c_raw_bb / length(train_idx);
    
    % Avref
    c_av_bb = (d_av * d_av') / size(d_av, 2);
    c_av_bb = real((c_av_bb + c_av_bb') / 2);
    if trace(c_av_bb) > 0, c_av_bb = c_av_bb * (n_tmpl / trace(c_av_bb)); end
    C_343_avref{1} = C_343_avref{1} + c_av_bb / length(train_idx);
    
    % Bands 2 to 9: Wavelet decomposition into 8 bands
    for wb = 1:n_wavelet_bands
        % Raw
        d_wb_raw = stateful_modwt_single_band(d_raw', 'haar', 7, wb)';
        c_wb_raw = (d_wb_raw * d_wb_raw') / size(d_wb_raw, 2);
        c_wb_raw = real((c_wb_raw + c_wb_raw') / 2);
        if trace(c_wb_raw) > 0, c_wb_raw = c_wb_raw * (n_tmpl / trace(c_wb_raw)); end
        C_343_raw{wb + 1} = C_343_raw{wb + 1} + c_wb_raw / length(train_idx);
        
        % Avref
        d_wb_av = stateful_modwt_single_band(d_av', 'haar', 7, wb)';
        c_wb_av = (d_wb_av * d_wb_av') / size(d_wb_av, 2);
        c_wb_av = real((c_wb_av + c_wb_av') / 2);
        if trace(c_wb_av) > 0, c_wb_av = c_wb_av * (n_tmpl / trace(c_wb_av)); end
        C_343_avref{wb + 1} = C_343_avref{wb + 1} + c_wb_av / length(train_idx);
    end
end

% Final symmetry and trace check
for b = 1:n_bands
    C_343_avref{b} = real((C_343_avref{b} + C_343_avref{b}') / 2);
    if trace(C_343_avref{b}) > 0
        C_343_avref{b} = C_343_avref{b} * (n_tmpl / trace(C_343_avref{b}));
    end
    
    C_343_raw{b} = real((C_343_raw{b} + C_343_raw{b}') / 2);
    if trace(C_343_raw{b}) > 0
        C_343_raw{b} = C_343_raw{b} * (n_tmpl / trace(C_343_raw{b}));
    end
    
    C_27_avref{b} = real((C_27_avref{b} + C_27_avref{b}') / 2);
    if trace(C_27_avref{b}) > 0
        C_27_avref{b} = C_27_avref{b} * (27 / trace(C_27_avref{b}));
    end
end

% Package into struct
GEDAI_networks = struct();
GEDAI_networks.C_343_avref = C_343_avref;
GEDAI_networks.C_343_raw   = C_343_raw;
GEDAI_networks.C_27_avref  = C_27_avref;
GEDAI_networks.chanlocs_27 = chanlocs_27;
GEDAI_networks.optimal_gammas = optimal_gammas;
GEDAI_networks.band_names  = band_names;
GEDAI_networks.train_subjects = train_idx;
GEDAI_networks.template_electrodes = {tmpl_locs.labels};
GEDAI_networks.description = 'Precomputed 343-channel empirical residual network reference covariances for GEDAI frequency-tuned deartifacting';
GEDAI_networks.date_created = datestr(now);

out_file = fullfile(aux_dir, 'GEDAI_empirical_residual_networks.mat');
fprintf('\nSaving empirical networks to: %s\n', out_file);
save(out_file, 'GEDAI_networks', '-v7');
fprintf('Done! Successfully created %s\n', out_file);
