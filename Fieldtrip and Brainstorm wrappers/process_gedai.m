function varargout = process_gedai( varargin )
% PROCESS_GEDAI: Wrapper for GEDAI.m function to be used in Brainstorm
%
% USAGE:                sProcess = process_gedai('GetDescription')
%                       OutputFiles = process_gedai('Run', sProcess, sInputs)

% [Generalized Eigenvalue De-Artifacting Instrument (GEDAI) v 1.7]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2026] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% For any questions, please contact:
% dr.t.ros@gmail.com
%
% Authors: Tomas Ros, Center for Biomedical Imaging (CIBM), University of Geneva, 2025

eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    sProcess.Comment     = 'GEDAI';
    sProcess.FileTag     = 'gedai';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'Artifacts';
    sProcess.Index       = 113.7;
    sProcess.InputTypes  = {'data', 'raw'};
    sProcess.OutputTypes = {'data', 'raw'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    sProcess.Description = 'https://neuroimage.usc.edu/brainstorm/Tutorials/Gedai';
    sProcess.isSeparator = 1;

    % === Artifact threshold type
    sProcess.options.label1.Comment = '<B>Artifact threshold type</B>';
    sProcess.options.label1.Type    = 'label';
    sProcess.options.artifact_threshold_type.Comment = {'auto- &nbsp', 'auto &nbsp', 'auto+ &nbsp', ''; ...
                                                        'auto-', 'auto', 'auto+', ''};
    sProcess.options.artifact_threshold_type.Type    = 'radio_linelabel';
    sProcess.options.artifact_threshold_type.Value   = 'auto';
    % === Epoch size in cycles
    sProcess.options.epoch_size_in_cycles.Comment = 'Epoch size in wave cycles (e.g., 12)';
    sProcess.options.epoch_size_in_cycles.Type    = 'value';
    sProcess.options.epoch_size_in_cycles.Value   = {12, 'cycles', 0};
    % === Low-cut frequency
    sProcess.options.lowcut_frequency.Comment = 'Low-cut frequency';
    sProcess.options.lowcut_frequency.Type    = 'value';
    sProcess.options.lowcut_frequency.Value   = {1.0, 'Hz', 1};
    % === Reference matrix type
    sProcess.options.label2.Comment = '<B>Leadfield matrix</B>';
    sProcess.options.label2.Type    = 'label';
    sProcess.options.ref_matrix_type.Comment = {'Freesurfer precomputed (for standard EEG electrode locations)', 'Freesurfer interpolated (for non-standard EEG electrode locations)', 'Brainstorm headmodel (custom for M/EEG)'; ...
                                                'fs_precomputed', 'fs_interpolated', 'bst_headmodel'};
    sProcess.options.ref_matrix_type.Type    = 'radio_label';
    sProcess.options.ref_matrix_type.Value   = 'bst_headmodel';
    % === Parallel processing
    sProcess.options.label3.Comment   = '<BR>';
    sProcess.options.label3.Type      = 'label';
    sProcess.options.parallel.Comment = 'Use parallel processing (N.B. needs a lot more RAM)';
    sProcess.options.parallel.Type    = 'checkbox';
    sProcess.options.parallel.Value   = 1;
    % === Visualize artifacts
    sProcess.options.visualize_artifacts.Comment = 'Visualize artifacts';
    sProcess.options.visualize_artifacts.Type    = 'checkbox';
    sProcess.options.visualize_artifacts.Value   = 1;
    % === Save artifacts data
    sProcess.options.save_artifacts.Comment = 'Save artifacts data';
    sProcess.options.save_artifacts.Type    = 'checkbox';
    sProcess.options.save_artifacts.Value   = 0;
    sProcess.isSeparator = 1;
    % === ENOVA bad epoch rejection
    sProcess.options.label4.Comment = '<B>ENOVA bad epoch rejection</B>';
    sProcess.options.label4.Type    = 'label';
    sProcess.options.reject_by_enova.Comment = 'Enable';
    sProcess.options.reject_by_enova.Type    = 'checkbox';
    sProcess.options.reject_by_enova.Value   = 0;
    sProcess.options.reject_by_enova.Controller = 'enova';
    sProcess.options.enova_threshold.Comment = 'ENOVA Threshold (0-1)';
    sProcess.options.enova_threshold.Type    = 'value';
    sProcess.options.enova_threshold.Value   = {0.9, '', 2};
    sProcess.options.enova_threshold.Class   = 'enova';
    sProcess.isSeparator = 1;
    % === ENOVA bad channel rejection
    sProcess.options.label5.Comment = '<B>ENOVA bad channel rejection</B>';
    sProcess.options.label5.Type    = 'label';
    sProcess.options.reject_channels_by_enova.Comment = 'Enable';
    sProcess.options.reject_channels_by_enova.Type    = 'checkbox';
    sProcess.options.reject_channels_by_enova.Value   = 0;
    sProcess.options.reject_channels_by_enova.Controller = 'enova_chan';
    sProcess.options.enova_threshold_per_channel.Comment = 'Channel ENOVA Threshold (0-1)';
    sProcess.options.enova_threshold_per_channel.Type    = 'value';
    sProcess.options.enova_threshold_per_channel.Value   = {0.9, '', 2};
    sProcess.options.enova_threshold_per_channel.Class   = 'enova_chan';
end


%% ===== GET OPTIONS =====
function [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, enova_threshold, enova_threshold_per_channel, save_artifacts] = GetOptions(sProcess)
    artifact_threshold_type = sProcess.options.artifact_threshold_type.Value;
    epoch_size_in_cycles    = sProcess.options.epoch_size_in_cycles.Value{1};
    lowcut_frequency        = sProcess.options.lowcut_frequency.Value{1};
    switch sProcess.options.ref_matrix_type.Value
        case 'fs_precomputed',  ref_matrix_type = 'Freesurfer (precomputed)';
        case 'fs_interpolated', ref_matrix_type = 'Freesurfer (interpolated)';
        case 'bst_headmodel',   ref_matrix_type = 'Brainstorm leadfield';
    end
    parallel             = sProcess.options.parallel.Value;
    visualize_artifacts  = sProcess.options.visualize_artifacts.Value;
    if isfield(sProcess.options, 'save_artifacts') && isfield(sProcess.options.save_artifacts, 'Value')
        save_artifacts = sProcess.options.save_artifacts.Value;
    else
        save_artifacts = 0;
    end
    if isfield(sProcess.options, 'reject_by_enova') && sProcess.options.reject_by_enova.Value
        enova_threshold = sProcess.options.enova_threshold.Value{1};
    else
        enova_threshold = [];
    end
    if isfield(sProcess.options, 'reject_channels_by_enova') && sProcess.options.reject_channels_by_enova.Value
        enova_threshold_per_channel = sProcess.options.enova_threshold_per_channel.Value{1};
    else
        enova_threshold_per_channel = [];
    end
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, ~, ~, enova_threshold, enova_threshold_per_channel, ~] = GetOptions(sProcess);
    Comment = ['GEDAI: ' artifact_threshold_type ', ' num2str(epoch_size_in_cycles) ' cycles, ' num2str(lowcut_frequency) ' Hz, ' ref_matrix_type];
    if ~isempty(enova_threshold)
        Comment = [Comment, ', ENOVA=' num2str(enova_threshold)];
    end
    if ~isempty(enova_threshold_per_channel)
        Comment = [Comment, ', ChENOVA=' num2str(enova_threshold_per_channel)];
    end
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>
    OutputFiles = {};

    % Check if GEDAI plugin is loaded
    PlugDesc = bst_plugin('GetDescription', 'gedai');
    if ~isequal(PlugDesc.isLoaded, 1) || isempty(PlugDesc.Path)
        [isOk, errMsg] = bst_plugin('Load', 'gedai');
        if ~isOk
            bst_report('Error', sProcess, sInputs, errMsg);
            return
        end
    end

    % Get options
    [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, enova_threshold, enova_threshold_per_channel, save_artifacts] = GetOptions(sProcess);

    % Iterate over inputs
    for iInput = 1:length(sInputs)
        sInput = sInputs(iInput);

        try
            % =========================================================
            % STEP 1: LOAD DATA
            % =========================================================
            % Flag for raw recordings
            isRaw = strcmpi(sInput.FileType, 'raw');
            % Get channel file
            [sChannel, iStudyChannel] = bst_get('ChannelForStudy', sInput.iStudy);
            ChannelMat = in_bst_channel(sChannel.FileName);
            channelFlag = [];
            % Load the BST file structure (works for both raw links and imported data)
            FileMat = in_bst_data(sInput.FileName);

            % Read all data in data file
            if isRaw
                % For raw links, FileMat.F is the sFile descriptor struct.
                % Use in_fread with the correct Brainstorm API.
                disp(['GEDAI> Reading raw file: ' sInput.FileName]);
                sFile = FileMat.F;
                % Force reading clean data: Apply CTF compensators and apply Projectors
                ImportOptions = db_template('ImportOptions');
                ImportOptions.ImportMode      = 'Time';
                ImportOptions.UseCtfComp      = 1;
                ImportOptions.UseSsp          = 1;
                ImportOptions.EventsMode      = 'ignore';
                ImportOptions.DisplayMessages = 0;
                ImportOptions.RemoveBaseline  = 'no';
                % Read all channels all samples, epoch 1
                [DataMatrix, TimeVector] = in_fread(sFile, ChannelMat, 1, [], [], ImportOptions);
                sInput.A          = DataMatrix;
                sInput.TimeVector = TimeVector;
                if isfield(sFile, 'channelflag'), channelFlag = sFile.channelflag; end
            else
                % Imported data: data matrix is directly in FileMat.F
                sInput.A          = FileMat.F;
                sInput.TimeVector = FileMat.Time;
                if isfield(FileMat, 'ChannelFlag'), channelFlag = FileMat.ChannelFlag; end
            end

            % =========================================================
            % STEP 2: CHANNEL SELECTION
            % =========================================================
            if isempty(channelFlag)
                channelFlag = ones(length(ChannelMat.Channel), 1);
            end
            eeg_meg_idx = find(ismember({ChannelMat.Channel.Type}, {'EEG', 'MEG', 'MEG MAG', 'MEG GRAD'}) & (channelFlag(:)' == 1));
            if isempty(eeg_meg_idx)
                bst_report('Error', sProcess, sInput, 'No EEG or MEG channels found.');
                continue;
            end

            ChannelMatFiltered = ChannelMat;
            ChannelMatFiltered.Channel = ChannelMat.Channel(eeg_meg_idx);

            channel_types = {ChannelMatFiltered.Channel.Type};
            eeg_count = sum(strcmp(channel_types, 'EEG'));
            meg_count = sum(ismember(channel_types, {'MEG', 'MEG MAG', 'MEG GRAD'}));

            if eeg_count > 0 && meg_count > 0
                bst_report('Error', sProcess, sInput, 'Cannot process mixed EEG and MEG channels. Please process them separately.');
                continue;
            elseif eeg_count > 0
                signal_type = 'eeg';
                process_mag_grad_separately = false;
            elseif meg_count > 0
                signal_type = 'meg';
                mag_count  = sum(strcmp(channel_types, 'MEG MAG'));
                grad_count = sum(strcmp(channel_types, 'MEG GRAD'));
                process_mag_grad_separately = (mag_count > 0 && grad_count > 0);
            else
                bst_report('Error', sProcess, sInput, 'No valid EEG or MEG channels detected.');
                continue;
            end

            % =========================================================
            % STEP 3: HEAD MODEL
            % =========================================================
            Gain_avref = [];
            if strcmp(ref_matrix_type, 'Brainstorm leadfield')
                HeadModelFile = [];
                sStudyData = bst_get('Study', sInput.iStudy);
                sStudyChan = bst_get('Study', iStudyChannel);

                if ~isempty(sStudyData.iHeadModel) && ~isempty(sStudyData.HeadModel)
                    HeadModelFile = sStudyData.HeadModel(sStudyData.iHeadModel).FileName;
                end
                if isempty(HeadModelFile) && ~isempty(sStudyChan.iHeadModel) && ~isempty(sStudyChan.HeadModel)
                    HeadModelFile = sStudyChan.HeadModel(sStudyChan.iHeadModel).FileName;
                end
                if isempty(HeadModelFile) && ~isempty(sStudyData.HeadModel)
                    HeadModelFile = sStudyData.HeadModel(1).FileName;
                end
                if isempty(HeadModelFile) && ~isempty(sStudyChan.HeadModel)
                    HeadModelFile = sStudyChan.HeadModel(1).FileName;
                end
                if isempty(HeadModelFile)
                    bst_report('Error', sProcess, sInput, 'No head model found.');
                    continue;
                end

                HeadModel    = in_bst_headmodel(HeadModelFile, 0, 'Gain');
                Gain_filtered = HeadModel.Gain(eeg_meg_idx, :);
                if strcmp(signal_type, 'eeg')
                    % Non-rank-deficient average reference (formula: G - sum(G)/(N+1))
                    Gain_avref = Gain_filtered - sum(Gain_filtered, 1) / (size(Gain_filtered, 1) + 1);
                else
                    Gain_avref = Gain_filtered;
                end
            end

            % =========================================================
            % STEP 4: RUN GEDAI
            % =========================================================
            if process_mag_grad_separately
                mag_idx_in_filtered  = find(strcmp(channel_types, 'MEG MAG'));
                grad_idx_in_filtered = find(strcmp(channel_types, 'MEG GRAD'));

                sInputFiltered   = sInput;
                sInputFiltered.A = sInput.A(eeg_meg_idx, :);

                % --- MAG ---
                ChannelMatMAG = ChannelMatFiltered;
                ChannelMatMAG.Channel = ChannelMatFiltered.Channel(mag_idx_in_filtered);
                sInputMAG   = sInputFiltered;
                sInputMAG.A = sInputFiltered.A(mag_idx_in_filtered, :);
                EEG_MAG = brainstorm2eeglab(sInputMAG, ChannelMatMAG);
                if length(sInputMAG.TimeVector) > 1, EEG_MAG.srate = 1 / mean(diff(sInputMAG.TimeVector)); end
                if strcmp(ref_matrix_type, 'Brainstorm leadfield')
                    Gain_MAG = Gain_avref(mag_idx_in_filtered, :);
                    ref_matrix_param_MAG = Gain_MAG * Gain_MAG';
                elseif strcmp(ref_matrix_type, 'Freesurfer (precomputed)')
                    ref_matrix_param_MAG = 'precomputed';
                else
                    ref_matrix_param_MAG = 'interpolated';
                end
                [EEGclean_MAG, EEGartifacts_MAG] = GEDAI(EEG_MAG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_MAG, parallel, visualize_artifacts, enova_threshold, enova_threshold_per_channel, signal_type);

                % --- GRAD ---
                ChannelMatGRAD = ChannelMatFiltered;
                ChannelMatGRAD.Channel = ChannelMatFiltered.Channel(grad_idx_in_filtered);
                sInputGRAD   = sInputFiltered;
                sInputGRAD.A = sInputFiltered.A(grad_idx_in_filtered, :);
                EEG_GRAD = brainstorm2eeglab(sInputGRAD, ChannelMatGRAD);
                if length(sInputGRAD.TimeVector) > 1, EEG_GRAD.srate = 1 / mean(diff(sInputGRAD.TimeVector)); end
                if strcmp(ref_matrix_type, 'Brainstorm leadfield')
                    Gain_GRAD = Gain_avref(grad_idx_in_filtered, :);
                    ref_matrix_param_GRAD = Gain_GRAD * Gain_GRAD';
                elseif strcmp(ref_matrix_type, 'Freesurfer (precomputed)')
                    ref_matrix_param_GRAD = 'precomputed';
                else
                    ref_matrix_param_GRAD = 'interpolated';
                end
                [EEGclean_GRAD, EEGartifacts_GRAD] = GEDAI(EEG_GRAD, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_GRAD, parallel, visualize_artifacts, enova_threshold, enova_threshold_per_channel, signal_type);

                target_length = size(sInputFiltered.A, 2);
                mask_MAG = get_samples_to_keep_mask(EEGclean_MAG, target_length);
                mask_GRAD = get_samples_to_keep_mask(EEGclean_GRAD, target_length);
                combined_mask = mask_MAG & mask_GRAD;

                [EEGclean_MAG, EEGartifacts_MAG] = align_gedai_outputs_to_mask(EEGclean_MAG, EEGartifacts_MAG, combined_mask, mask_MAG, 'MAG');
                [EEGclean_GRAD, EEGartifacts_GRAD] = align_gedai_outputs_to_mask(EEGclean_GRAD, EEGartifacts_GRAD, combined_mask, mask_GRAD, 'GRAD');

                % --- Recombine ---
                EEGclean = brainstorm2eeglab(sInputFiltered, ChannelMatFiltered);
                EEGclean.data = EEGclean.data(:, combined_mask);
                EEGclean.pnts = size(EEGclean.data, 2);
                EEGclean.xmax = EEGclean.xmin + (EEGclean.pnts - 1) / EEGclean.srate;
                if EEGclean.pnts > 1
                    EEGclean.times = linspace(EEGclean.xmin * 1000, EEGclean.xmax * 1000, EEGclean.pnts);
                else
                    EEGclean.times = EEGclean.xmin * 1000;
                end
                EEGclean.data(mag_idx_in_filtered, :)  = EEGclean_MAG.data;
                EEGclean.data(grad_idx_in_filtered, :) = EEGclean_GRAD.data;
                EEGartifacts = EEGclean;
                EEGartifacts.data(mag_idx_in_filtered, :)  = EEGartifacts_MAG.data;
                EEGartifacts.data(grad_idx_in_filtered, :) = EEGartifacts_GRAD.data;

                % Combine rejection masks
                if isfield(EEGclean_MAG.etc, 'GEDAI')
                    EEGclean.etc.GEDAI = EEGclean_MAG.etc.GEDAI;
                elseif isfield(EEGclean_GRAD.etc, 'GEDAI')
                    EEGclean.etc.GEDAI = EEGclean_GRAD.etc.GEDAI;
                end
                EEGclean.etc.GEDAI.samples_to_keep = combined_mask;
                EEGclean.etc.GEDAI.percentage_rejected = 100 * sum(~combined_mask) / length(combined_mask);

            else
                % Single processing (EEG or generic MEG)
                sInputFiltered   = sInput;
                sInputFiltered.A = sInput.A(eeg_meg_idx, :);
                EEG = brainstorm2eeglab(sInputFiltered, ChannelMatFiltered);
                if length(sInputFiltered.TimeVector) > 1
                    EEG.srate = 1 / mean(diff(sInputFiltered.TimeVector));
                end
                if strcmp(ref_matrix_type, 'Brainstorm leadfield')
                    ref_matrix_param = Gain_avref * Gain_avref';
                elseif strcmp(ref_matrix_type, 'Freesurfer (precomputed)')
                    ref_matrix_param = 'precomputed';
                else
                    ref_matrix_param = 'interpolated';
                end
                [EEGclean, EEGartifacts] = GEDAI(EEG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param, parallel, visualize_artifacts, enova_threshold, enova_threshold_per_channel, signal_type);
            end

            % =========================================================
            % STEP 5: BUILD OUTPUT STRUCTURES
            % =========================================================
            % Compute comment strings FIRST (used by both Cleaned and Artifacts)
            current_comment = sInput.Comment;
            % Remove time pattern from name. E.g. "Raw (0.00s,59.99s)" -- > Raw
            current_comment = regexprep(current_comment, ' \([\d\.]+s,[\d\.]+s\)', '');
            gedai_params    = FormatComment(sProcess);

            % Apply rejection mask to the full data matrix
            DataOut = sInput.A;
            TimeOut = sInput.TimeVector;
            if isfield(EEGclean.etc, 'GEDAI') && isfield(EEGclean.etc.GEDAI, 'samples_to_keep')
                mask = EEGclean.etc.GEDAI.samples_to_keep;
                if length(mask) == size(DataOut, 2)
                    DataOut = DataOut(:, mask);
                    if length(TimeOut) == length(mask)
                        TimeOut = TimeOut(mask);
                    end
                end
            end

            % Replace EEG/MEG channels with cleaned data.
            % EEGclean.data is in µV (brainstorm2eeglab multiplied by 1e6),
            % so divide back to Volts for Brainstorm storage.
            DataOut(eeg_meg_idx, :) = EEGclean.data / 1e6;

            new_duration = TimeOut(end) - TimeOut(1);
            if isfield(EEGclean.etc, 'GEDAI')
                rej_percent  = EEGclean.etc.GEDAI.percentage_rejected;
                gedai_params = [gedai_params, sprintf(', Rej=%.1f%% (%.1fs)', rej_percent, new_duration)];
            end

            % Build ChannelFlag (must be nChannels x 1)
            nChannelsTotal = size(DataOut, 1);
            if length(channelFlag) == nChannelsTotal
                ChannelFlagOut = channelFlag(:);
            else
                ChannelFlagOut = ones(nChannelsTotal, 1);
            end

            % --- Cleaned file ---
            FileMatCleaned             = FileMat;
            FileMatCleaned.Comment     = ['Cleaned | ', current_comment, ' | ', gedai_params];
            sInput.A                   = DataOut;
            FileMatCleaned.Time        = TimeOut;
            FileMatCleaned.ChannelFlag = ChannelFlagOut;
            OutputFiles{end+1} = SaveAndRegisterData(sInput, 'cleaned', FileMatCleaned, ChannelMat);

            % --- Artifacts file ---
            if save_artifacts
                try
                    % Artifact data: zeros for non-EEG/MEG, artifact signal for EEG/MEG
                    % Convert artifact data from µV back to Volts
                    ArtifactData = zeros(nChannelsTotal, size(DataOut, 2));
                    if size(EEGartifacts.data, 2) == size(DataOut, 2)
                        ArtifactData(eeg_meg_idx, :) = EEGartifacts.data / 1e6;
                    else
                        warning('GEDAI:ArtifactDimensionMismatch', 'Artifact time dimension does not match cleaned data.');
                    end

                    FileMatArtifacts             = FileMat;
                    FileMatArtifacts.Comment     = ['Artifacts | ', current_comment, ' | ', gedai_params];
                    sInput.A                     = ArtifactData;
                    FileMatArtifacts.Time        = TimeOut;
                    FileMatArtifacts.ChannelFlag = ChannelFlagOut;
                    OutputFiles{end+1} = SaveAndRegisterData(sInput, 'artifacts', FileMatArtifacts, ChannelMat);

                catch ME_Art
                    warning('GEDAI:ArtifactSaveFailed', 'Failed to save Artifacts file: %s', ME_Art.message);
                    disp(getReport(ME_Art));
                end
            end

        catch ME
            bst_report('Error', sProcess, sInput, ['GEDAI Failed: ' ME.message]);
            disp(getReport(ME));
        end
    end
end


%% ===== HELPER FUNCTIONS =====
function EEG = brainstorm2eeglab(sInput, ChannelMat)
    EEG.setname  = sInput.Comment;
    EEG.filename = sInput.FileName;
    EEG.filepath = fileparts(sInput.FileName);
    EEG.subject  = '';
    EEG.group    = '';
    EEG.condition = '';
    EEG.session  = [];
    EEG.nbchan   = size(sInput.A, 1);
    EEG.trials   = 1;
    EEG.pnts     = size(sInput.A, 2);
    EEG.srate    = 1 / (sInput.TimeVector(2) - sInput.TimeVector(1));
    EEG.xmin     = sInput.TimeVector(1);
    EEG.xmax     = sInput.TimeVector(end);
    EEG.times    = sInput.TimeVector * 1000; % ms
    EEG.data     = sInput.A * 1e6;           % V -> µV
    EEG.etc      = [];
    EEG.event    = [];
    EEG.icaweights = [];
    EEG.icasphere  = [];
    EEG.icawinv    = [];
    EEG.icaact     = [];

    for i = 1:length(ChannelMat.Channel)
        EEG.chanlocs(i).labels = ChannelMat.Channel(i).Name;
        if ~isempty(ChannelMat.Channel(i).Loc)
            EEG.chanlocs(i).X = ChannelMat.Channel(i).Loc(1) * 1000;
            EEG.chanlocs(i).Y = ChannelMat.Channel(i).Loc(2) * 1000;
            EEG.chanlocs(i).Z = ChannelMat.Channel(i).Loc(3) * 1000;
        else
            EEG.chanlocs(i).X = NaN;
            EEG.chanlocs(i).Y = NaN;
            EEG.chanlocs(i).Z = NaN;
        end
        EEG.chanlocs(i).type = ChannelMat.Channel(i).Type;
    end

    % Ensure all standard Cartesian, polar, and spherical fields exist and are populated
    try
        EEG.chanlocs = convertlocs(EEG.chanlocs, 'cart2all');
    catch
        % If convertlocs fails or is missing, initialize them manually to prevent crashes
        for i = 1:length(EEG.chanlocs)
            if ~isfield(EEG.chanlocs(i), 'theta') || isempty(EEG.chanlocs(i).theta)
                x = EEG.chanlocs(i).X; y = EEG.chanlocs(i).Y; z = EEG.chanlocs(i).Z;
                if ~isnan(x) && ~isnan(y) && ~isnan(z)
                    r = sqrt(x^2 + y^2 + z^2);
                    if r > 0
                        EEG.chanlocs(i).theta = -atan2d(x, y);
                        EEG.chanlocs(i).radius = sqrt(x^2 + y^2) / r * 0.5;
                    else
                        EEG.chanlocs(i).theta = 0;
                        EEG.chanlocs(i).radius = 0;
                    end
                else
                    EEG.chanlocs(i).theta = 0;
                    EEG.chanlocs(i).radius = 0;
                end
            end
            if ~isfield(EEG.chanlocs(i), 'sph_theta'),   EEG.chanlocs(i).sph_theta = []; end
            if ~isfield(EEG.chanlocs(i), 'sph_phi'),     EEG.chanlocs(i).sph_phi = []; end
            if ~isfield(EEG.chanlocs(i), 'sph_radius'),  EEG.chanlocs(i).sph_radius = []; end
            if ~isfield(EEG.chanlocs(i), 'urchan'),      EEG.chanlocs(i).urchan = i; end
            if ~isfield(EEG.chanlocs(i), 'ref'),         EEG.chanlocs(i).ref = ''; end
        end
    end
end

function mask = get_samples_to_keep_mask(EEG, target_length)
    mask = true(1, target_length);
    if isfield(EEG, 'etc') && isfield(EEG.etc, 'GEDAI') && isfield(EEG.etc.GEDAI, 'samples_to_keep')
        candidate_mask = logical(EEG.etc.GEDAI.samples_to_keep(:)');
        if length(candidate_mask) == target_length
            mask = candidate_mask;
        else
            warning('GEDAI:MaskLengthMismatch', 'Ignoring a GEDAI samples_to_keep mask with unexpected length.');
        end
    end
end

function [EEGclean, EEGartifacts] = align_gedai_outputs_to_mask(EEGclean, EEGartifacts, target_mask, source_mask, sensor_label)
    if length(source_mask) ~= length(target_mask)
        error('GEDAI:%sMaskMismatch', sensor_label, 'Sample masks must have the same length for recombination.');
    end

    selection_mask = target_mask(source_mask);
    expected_columns = sum(source_mask);
    if size(EEGclean.data, 2) ~= expected_columns || size(EEGartifacts.data, 2) ~= expected_columns
        error('GEDAI:%sOutputMismatch', sensor_label, 'GEDAI output length does not match its rejection mask for %s sensors.', sensor_label);
    end

    EEGclean.data = EEGclean.data(:, selection_mask);
    EEGartifacts.data = EEGartifacts.data(:, selection_mask);

    EEGclean.pnts = size(EEGclean.data, 2);
    EEGartifacts.pnts = size(EEGartifacts.data, 2);
    EEGclean.xmax = EEGclean.xmin + (EEGclean.pnts - 1) / EEGclean.srate;
    EEGartifacts.xmax = EEGartifacts.xmin + (EEGartifacts.pnts - 1) / EEGartifacts.srate;
    if EEGclean.pnts > 1
        EEGclean.times = linspace(EEGclean.xmin * 1000, EEGclean.xmax * 1000, EEGclean.pnts);
        EEGartifacts.times = linspace(EEGartifacts.xmin * 1000, EEGartifacts.xmax * 1000, EEGartifacts.pnts);
    else
        EEGclean.times = EEGclean.xmin * 1000;
        EEGartifacts.times = EEGartifacts.xmin * 1000;
    end
end

function OutputFile = SaveAndRegisterData(sInput, GedaiDataType, FileMatOut, ChannelMat)
    % DataOut is the field A in sInput
    DataOut = sInput.A;

    if strcmpi(sInput.FileType, 'raw')
        % Save as raw file, in its own raw Study
        ConditionName = [sInput.Condition, '_gedai_' GedaiDataType];
        ProtocolInfo = bst_get('ProtocolInfo');
        pathStudyOut = file_unique(bst_fullfile(ProtocolInfo.STUDIES, sInput.SubjectName, ConditionName));
        [~, ConditionName] = bst_fileparts(pathStudyOut, 1);
        % Output file name (derived from the condition name)
        [~, rawBaseOut] = bst_fileparts(pathStudyOut);
        rawBaseOut = regexprep(rawBaseOut, '^@raw', '');
        RawFileOut = bst_fullfile(pathStudyOut, [rawBaseOut '.bst']);
        % Get input study (to copy the creation date)
        sInputStudy = bst_get('AnyFile', sInput.FileName);
        % Create output study
        iStudyOut = db_add_condition(sInput.SubjectName, ConditionName, [], sInputStudy.DateOfStudy);
        sStudyOut = bst_get('Study', iStudyOut);
        % Full name for link-to-raw file
        MatFileName = bst_fullfile(ProtocolInfo.STUDIES, bst_fileparts(sStudyOut.FileName), ['data_0raw_' rawBaseOut '.mat']);
        % Create empty Brainstorm-binary file
        sFileOut = out_fopen(RawFileOut, 'BST-BIN', FileMatOut.F, ChannelMat);
        % Update sFileOut
        sFileOut.prop.times = FileMatOut.Time([1,end]);
        % Set Output sFile structure
        FileMatOut.F = sFileOut;
        bst_save(MatFileName, FileMatOut, 'v6');
        % Create new channel file
        db_set_channel(iStudyOut, ChannelMat, 2, 0);
        % Write all data in one block
        out_fwrite(sFileOut, ChannelMat, 1, [], [], DataOut);
        db_add_data(iStudyOut, MatFileName, FileMatOut);
        OutputFile = MatFileName;
    else
        % Save as imported data, use input study folder
        sStudyOut = bst_get('Study', sInput.iStudy);
        pathStudyOut = bst_fileparts(file_fullpath(sStudyOut.FileName));
        MatFileName = bst_process('GetNewFilename', pathStudyOut, ['data_gedai_' GedaiDataType]);
        % Update FileMatOut
        FileMatOut.F = DataOut;
        bst_save(MatFileName, FileMatOut, 'v6');
        db_add_data(sInput.iStudy, MatFileName, FileMatOut);
        OutputFile = MatFileName;
    end
end

