function [data_clean, data_artifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band] = ft_denoise_gedai(cfg, data)
% ft_denoise_gedai  FieldTrip wrapper for the GEDAI plugin.
%
% Usage:
%   [data_clean, data_artifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band] = ft_denoise_gedai(cfg, data)
%   data_clean = ft_denoise_gedai(cfg, data)
%
% Required inputs:
%   cfg.dataset               (char)    - Path to the original dataset file
%   data                      (struct)  - FieldTrip data structure (ft_read_data output)
%
% Optional cfg fields:
%   cfg.epoch_size               (double)  - Epoch size in wave CYCLES. [Default: 12]
%   cfg.lowcut_frequency         (double)  - Low-cut frequency in Hz. [Default: 0.5]
%   cfg.ref_matrix_type          (char or matrix) - Leadfield reference. [Default: 'precomputed']
%                                   Matrix used as a reference for deartifacting.
%                                  The default "precomputed" uses a BEM leadfield for
%                                  standard electrode locations precomputed through 
%                                  OPENMEEG (343 electrodes) based on 10-5 system. 
%                                 "interpolated" uses the precomputed leadfield and 
%                                 interpolates it to non-standard electrode locations.
%                                 Altenatively, you can input a "custom" covariance matrix
%                                 (with dimensions channel x channel) using the name of its matlab variable
%   cfg.parallel                 (logical) - Use parallel processing. [Default: true]
%   cfg.visualize_artifacts      (logical) - Visualize artifacts. [Default: false]
%   cfg.visualize_manifold       (logical) - Visualize SENSAI manifold. [Default: false]
%   cfg.cat_trials               (logical) - Concatenate trials before denoising. [Default: true]
%                                 Cleaned/artifact outputs are restored to the original
%                                 trial boundaries after denoising. Scalar summary metrics
%                                 remain aggregated across the concatenated run.
%   cfg.signal_type              (char)    - 'eeg' or 'meg'. Auto-detected if not set.
%
% Outputs:
%   data_clean                - FieldTrip struct, cleaned data
%   data_artifacts            - FieldTrip struct, artifact signal
%   SENSAI_score              - Per-trial SENSAI scores (cell)
%   SENSAI_score_per_band     - Per-trial per-band SENSAI scores (cell)
%   artifact_threshold_per_band - Per-trial per-band thresholds (cell)
%
% Notes:
%   - For MEG data with both magnetometers (MEG MAG) and gradiometers (MEG GRAD),
%     GEDAI is run separately on each sensor type and the results are recombined.
%     This mirrors the behaviour of the Brainstorm wrapper (process_gedai.m).
%   - signal_type is auto-detected from data.grad / data.elec / ft_chantype.
%     Set cfg.signal_type explicitly to override.

% Author: Yingqi Huang & Tomas Ros, University of Geneva, 2025-2026
% Updated: auto signal-type detection and MAG/GRAD split processing

% ---------- input validation ----------
if nargin < 2 || ~isstruct(cfg) || ~isstruct(data)
    error('Usage: [data_clean,...] = ft_denoise_gedai(cfg, data)');
end

% ---------- defaults ----------
def.epoch_size              = 12;       % wave cycles (= epoch_size_in_cycles in GEDAI)
def.lowcut_frequency        = 0.5;
def.ref_matrix_type         = 'precomputed';
def.parallel                = true;
def.visualize_artifacts     = false;
def.visualize_manifold      = false;
def.enova_threshold_per_epoch = [];
def.enova_threshold_per_channel = [];
def.cat_trials              = true;
def.signal_type             = '';       % empty = auto-detect
def.smoothing_window_seconds = Inf;

cfg = applyDefaults(cfg, def);

% ---------- FieldTrip field checks ----------
reqFields = {'trial', 'time', 'label', 'fsample'};
for fi = reqFields
    if ~isfield(data, fi{1})
        error('Input "data" lacks required FieldTrip field: %s', fi{1});
    end
end
if ~isfield(cfg, 'dataset')
    error('cfg.dataset is required (path to the original data file).');
end
if ~iscell(data.trial) || ~iscell(data.time)
    error('FieldTrip data.trial and data.time must be cell arrays.');
end
if numel(data.trial) ~= numel(data.time)
    error('data.trial and data.time must have the same number of trials.');
end

% ---------- auto-detect signal type ----------
if isempty(cfg.signal_type)
    cfg.signal_type = detectSignalType(data);
    disp(['ft_denoise_gedai> Auto-detected signal type: ' cfg.signal_type]);
else
    cfg.signal_type = lower(cfg.signal_type);
end

% ---------- detect MAG / GRAD split for MEG ----------
process_mag_grad_separately = false;
mag_idx = [];
grad_idx = [];

if strcmp(cfg.signal_type, 'meg')
    chantypes = ft_chantype(data);
    mag_idx  = find(ismember(chantypes, {'megmag'}));
    grad_idx = find(ismember(chantypes, {'meggrad', 'megplanar'}));
    if ~isempty(mag_idx) && ~isempty(grad_idx)
        process_mag_grad_separately = true;
        disp('ft_denoise_gedai> Mixed MAG+GRAD detected: processing sensor types separately.');
    end
end

% ---------- concatenate trials if requested ----------
data_original = data;
if cfg.cat_trials
    nTrials   = numel(data.trial);
    nChan     = numel(data.label);
    trial_lengths = cellfun(@(trial) size(trial, 2), data.trial);
    total_pts = sum(trial_lengths);
    data_concat = zeros(nChan, total_pts, 'like', data.trial{1});
    sample_interval = 1 / data.fsample;
    first_trial_time = data.time{1};
    time_concat = first_trial_time(1) + (0:(total_pts - 1)) * sample_interval;
    idx_start = 1;
    for i = 1:nTrials
        current_len = trial_lengths(i);
        if size(data.trial{i}, 1) ~= nChan
            error('All trials must contain the same number of channels when cfg.cat_trials is true.');
        end
        idx_end = idx_start + current_len - 1;
        data_concat(:, idx_start:idx_end) = data.trial{i};
        idx_start = idx_end + 1;
    end
    data.trial = {data_concat};
    data.time  = {time_concat};
end

% ---------- process each trial ----------
if cfg.cat_trials
    [clean_concat, ~, art_concat, Sscore, Sband, artThr, sample_mask] = ...
        do_gedai(data, 1, cfg, process_mag_grad_separately, mag_idx, grad_idx);
    [data_clean, data_artifacts] = split_concatenated_trials(data_original, clean_concat, art_concat, sample_mask);
    SENSAI_score = Sscore;
    SENSAI_score_per_band = Sband;
    artifact_threshold_per_band = artThr;
else
    data_clean     = data;
    data_artifacts = data;
    SENSAI_score                = cell(numel(data.trial), 1);
    SENSAI_score_per_band       = cell(numel(data.trial), 1);
    artifact_threshold_per_band = cell(numel(data.trial), 1);

    nTr = numel(data.trial);
    for t = 1:nTr
        [data_clean.trial{t}, data_clean.time{t}, EEGart, Sscore, Sband, artThr] = ...
            do_gedai(data, t, cfg, process_mag_grad_separately, mag_idx, grad_idx);
        data_artifacts.trial{t}        = EEGart;
        SENSAI_score{t}                = Sscore;
        SENSAI_score_per_band{t}       = Sband;
        artifact_threshold_per_band{t} = artThr;
    end

    % Unwrap single-trial outputs to plain values for convenience
    if nTr == 1
        SENSAI_score                = SENSAI_score{1};
        SENSAI_score_per_band       = SENSAI_score_per_band{1};
        artifact_threshold_per_band = artifact_threshold_per_band{1};
    end
end


% ==========================================================================
%  SUBFUNCTIONS
% ==========================================================================

% ---------- do_gedai: run GEDAI on one trial ----------
function [dataClean, timeClean, artifactData, Sscore, Sband, Athr, sampleMask] = ...
        do_gedai(data, t, cfg, process_mag_grad_separately, mag_idx, grad_idx)

    trial_data = data.trial{t};   % [nChan x nTime]
    trial_time = data.time{t};

    if process_mag_grad_separately
        % --- Run GEDAI separately on MAG and GRAD ---
        EEGin_MAG  = buildEEGin(trial_data(mag_idx, :),  trial_time, data.label(mag_idx),  cfg, data);
        EEGin_GRAD = buildEEGin(trial_data(grad_idx, :), trial_time, data.label(grad_idx), cfg, data);

        [EEGclean_MAG,  EEGart_MAG,  Sscore_MAG,  Sband_MAG,  Athr_MAG]  = runGEDAI(EEGin_MAG,  cfg);
        [EEGclean_GRAD, EEGart_GRAD, Sscore_GRAD, Sband_GRAD, Athr_GRAD] = runGEDAI(EEGin_GRAD, cfg);

        mask_MAG = get_samples_to_keep_mask(EEGclean_MAG, numel(trial_time));
        mask_GRAD = get_samples_to_keep_mask(EEGclean_GRAD, numel(trial_time));
        combined_mask = mask_MAG & mask_GRAD;
        sampleMask = combined_mask;

        [clean_MAG, art_MAG] = align_outputs_to_mask(EEGclean_MAG.data, EEGart_MAG.data, combined_mask, mask_MAG, 'MAG');
        [clean_GRAD, art_GRAD] = align_outputs_to_mask(EEGclean_GRAD.data, EEGart_GRAD.data, combined_mask, mask_GRAD, 'GRAD');

        % --- Recombine into full channel matrix ---
        nChan = size(trial_data, 1);
        nTimeOut = sum(combined_mask);
        dataClean    = zeros(nChan, nTimeOut, 'like', trial_data);
        artifactData = zeros(nChan, nTimeOut, 'like', trial_data);
        dataClean(mag_idx, :)  = clean_MAG;
        dataClean(grad_idx, :) = clean_GRAD;
        artifactData(mag_idx, :)  = art_MAG;
        artifactData(grad_idx, :) = art_GRAD;

        timeClean = trial_time(combined_mask);

        % Merge per-band outputs
        Sscore = struct('MAG', Sscore_MAG, 'GRAD', Sscore_GRAD);
        Sband  = struct('MAG', Sband_MAG,  'GRAD', Sband_GRAD);
        Athr   = struct('MAG', Athr_MAG,   'GRAD', Athr_GRAD);

    else
        % --- Single pass (EEG or homogeneous MEG) ---
        EEGin = buildEEGin(trial_data, trial_time, data.label, cfg, data);
        [EEGclean, EEGart, Sscore, Sband, Athr] = runGEDAI(EEGin, cfg);

        mask = get_samples_to_keep_mask(EEGclean, numel(trial_time));
        sampleMask = mask;
        [dataClean, artifactData] = align_outputs_to_mask(EEGclean.data, EEGart.data, mask, mask, 'single-sensor');
        timeClean = trial_time(mask);
    end
end

function [data_clean_out, data_artifacts_out] = split_concatenated_trials(data_template, clean_concat, art_concat, sample_mask)
    data_clean_out = data_template;
    data_artifacts_out = data_template;

    sample_offset = 1;
    kept_offset = 1;
    total_kept = sum(sample_mask);
    for trial_idx = 1:numel(data_template.trial)
        current_len = size(data_template.trial{trial_idx}, 2);
        current_mask = sample_mask(sample_offset:sample_offset + current_len - 1);
        kept_len = sum(current_mask);

        if kept_len > 0
            kept_range = kept_offset:kept_offset + kept_len - 1;
            data_clean_out.trial{trial_idx} = clean_concat(:, kept_range);
            data_artifacts_out.trial{trial_idx} = art_concat(:, kept_range);
            kept_offset = kept_offset + kept_len;
        else
            data_clean_out.trial{trial_idx} = clean_concat(:, []);
            data_artifacts_out.trial{trial_idx} = art_concat(:, []);
        end
        data_clean_out.time{trial_idx} = data_template.time{trial_idx}(current_mask);
        data_artifacts_out.time{trial_idx} = data_template.time{trial_idx}(current_mask);
        sample_offset = sample_offset + current_len;
    end

    if kept_offset - 1 ~= total_kept || size(clean_concat, 2) ~= total_kept || size(art_concat, 2) ~= total_kept
        error('ft_denoise_gedai:SplitMismatch', 'Concatenated outputs could not be split back to the original trial boundaries.');
    end
end

function [EEGclean, EEGart, Sscore, Sband, Athr] = runGEDAI(EEGin, cfg)
    [EEGclean, EEGart, Sscore, Sband, Athr] = GEDAI( ...
        EEGin, ...
        cfg.epoch_size, ...
        cfg.lowcut_frequency, ...
        cfg.ref_matrix_type, ...
        cfg.parallel, ...
        cfg.visualize_artifacts || cfg.visualize_manifold, ...
        cfg.enova_threshold_per_epoch, ...
        cfg.enova_threshold_per_channel, ...
        cfg.signal_type, ...
        cfg.smoothing_window_seconds, ...
        '');
end

function mask = get_samples_to_keep_mask(EEG, target_length)
    mask = true(1, target_length);
    if isfield(EEG, 'etc') && isfield(EEG.etc, 'GEDAI') && isfield(EEG.etc.GEDAI, 'samples_to_keep')
        candidate_mask = logical(EEG.etc.GEDAI.samples_to_keep(:)');
        if length(candidate_mask) == target_length
            mask = candidate_mask;
        end
    end
end

function [clean_data, artifact_data] = align_outputs_to_mask(clean_data_in, artifact_data_in, target_mask, source_mask, sensor_label)
    if length(target_mask) ~= length(source_mask)
        error('ft_denoise_gedai:%sMaskMismatch', sensor_label, 'Target and source masks must have the same length.');
    end

    expected_columns = sum(source_mask);
    if size(clean_data_in, 2) ~= expected_columns || size(artifact_data_in, 2) ~= expected_columns
        error('ft_denoise_gedai:%sOutputMismatch', sensor_label, 'GEDAI output length does not match its keep-mask for %s data.', sensor_label);
    end

    selection_mask = target_mask(source_mask);
    clean_data = clean_data_in(:, selection_mask);
    artifact_data = artifact_data_in(:, selection_mask);
end

% ---------- buildEEGin: construct EEGLAB-style struct from FieldTrip data ----------
function EEGin = buildEEGin(trial_data, trial_time, labels, cfg, data)
    EEGin          = struct();
    EEGin.data     = trial_data;
    EEGin.srate    = data.fsample;
    EEGin.nbchan   = size(trial_data, 1);
    EEGin.pnts     = size(trial_data, 2);
    EEGin.times    = trial_time * 1000;  % s -> ms (EEGLAB convention)
    EEGin.xmin     = trial_time(1);
    EEGin.xmax     = trial_time(end);
    EEGin.trials   = 1;

    % Try to extract spatial coordinates from FieldTrip's elec/grad structure
    has_coords = false;
    loc_struct = [];
    if isfield(data, 'elec') && ~isempty(data.elec)
        loc_struct = data.elec;
    elseif isfield(data, 'grad') && ~isempty(data.grad)
        loc_struct = data.grad;
    end
    
    if ~isempty(loc_struct) && isfield(loc_struct, 'chanpos') && isfield(loc_struct, 'label')
        [found, loc_idx] = ismember(labels, loc_struct.label);
        if all(found)
            has_coords = true;
            chanpos = loc_struct.chanpos(loc_idx, :);
        end
    end

    chanlocs = struct('labels', labels(:)');
    for i = 1:numel(labels)
        chanlocs(i).type = cfg.signal_type;
        if has_coords
            chanlocs(i).X = chanpos(i, 1);
            chanlocs(i).Y = chanpos(i, 2);
            chanlocs(i).Z = chanpos(i, 3);
        else
            chanlocs(i).X = NaN;
            chanlocs(i).Y = NaN;
            chanlocs(i).Z = NaN;
        end
    end
    EEGin.chanlocs = chanlocs;

    % Ensure all standard Cartesian, polar, and spherical fields exist and are populated
    try
        EEGin.chanlocs = convertlocs(EEGin.chanlocs, 'cart2all');
    catch
        % If convertlocs fails or is missing, initialize them manually to prevent crashes
        for i = 1:length(EEGin.chanlocs)
            if ~isfield(EEGin.chanlocs(i), 'theta') || isempty(EEGin.chanlocs(i).theta)
                x = EEGin.chanlocs(i).X; y = EEGin.chanlocs(i).Y; z = EEGin.chanlocs(i).Z;
                if ~isnan(x) && ~isnan(y) && ~isnan(z)
                    r = sqrt(x^2 + y^2 + z^2);
                    if r > 0
                        EEGin.chanlocs(i).theta = -atan2d(x, y);
                        EEGin.chanlocs(i).radius = sqrt(x^2 + y^2) / r * 0.5;
                    else
                        EEGin.chanlocs(i).theta = 0;
                        EEGin.chanlocs(i).radius = 0;
                    end
                else
                    EEGin.chanlocs(i).theta = 0;
                    EEGin.chanlocs(i).radius = 0;
                end
            end
            if ~isfield(EEGin.chanlocs(i), 'sph_theta'),   EEGin.chanlocs(i).sph_theta = []; end
            if ~isfield(EEGin.chanlocs(i), 'sph_phi'),     EEGin.chanlocs(i).sph_phi = []; end
            if ~isfield(EEGin.chanlocs(i), 'sph_radius'),  EEGin.chanlocs(i).sph_radius = []; end
            if ~isfield(EEGin.chanlocs(i), 'urchan'),      EEGin.chanlocs(i).urchan = i; end
            if ~isfield(EEGin.chanlocs(i), 'ref'),         EEGin.chanlocs(i).ref = ''; end
        end
    end
    EEGin.etc      = struct();
    EEGin.event    = [];
    EEGin.epoch    = [];
    EEGin.reject   = struct();
    EEGin.stats    = struct();
    EEGin.specdata = [];
    EEGin.specicaact = [];
    EEGin.saved    = 'no';
    EEGin.history  = 'EEGin created in ft_denoise_gedai wrapper';
    EEGin.subject  = '';
    EEGin.group    = '';
    EEGin.condition = '';
    EEGin.ref      = 'average';
    EEGin.chaninfo = struct();
    EEGin.comments = 'Created by ft_denoise_gedai wrapper';
    [EEGin.filepath, EEGin.filename, ~] = fileparts(cfg.dataset);
end

% ---------- detectSignalType: infer 'eeg' or 'meg' from data struct ----------
function signal_type = detectSignalType(data)
    if isfield(data, 'grad') && ~isfield(data, 'elec')
        signal_type = 'meg';
    elseif isfield(data, 'elec') && ~isfield(data, 'grad')
        signal_type = 'eeg';
    else
        % Fallback: inspect channel type labels
        try
            chantypes = ft_chantype(data);
            if any(ismember(chantypes, {'megmag', 'meggrad', 'megplanar', 'meg'}))
                signal_type = 'meg';
            else
                signal_type = 'eeg';
            end
        catch
            warning('ft_denoise_gedai:signalTypeUnknown', ...
                'Could not auto-detect signal type. Defaulting to ''eeg''. Set cfg.signal_type explicitly.');
            signal_type = 'eeg';
        end
    end
end

% ---------- applyDefaults ----------
function cfg2 = applyDefaults(cfg1, def1)
    cfg2 = def1;
    fns  = fieldnames(cfg1);
    for k = 1:numel(fns)
        cfg2.(fns{k}) = cfg1.(fns{k});
    end
end

end
