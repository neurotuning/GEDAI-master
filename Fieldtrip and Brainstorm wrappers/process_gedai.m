function varargout = process_gedai( varargin )
% PROCESS_GEDAI: Wrapper for GEDAI.m function to be used in Brainstorm
%
% USAGE:                sProcess = process_gedai('GetDescription')
%                       OutputFiles = process_gedai('Run', sProcess, sInputs)

% [Generalized Eigenvalue De-Artifacting Instrument (GEDAI)]
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
    sProcess.options.lowcut_frequency.Value   = {0.5, 'Hz', 1};
    % === Reference matrix type
    sProcess.options.label2.Comment = '<B>Leadfield matrix</B>';
    sProcess.options.label2.Type    = 'label';
    sProcess.options.ref_matrix_type.Comment = {'Auto (Template anchor -> Empirical fallback)', 'Freesurfer precomputed (standard locations)', 'Freesurfer interpolated (non-standard locations)', 'Brainstorm headmodel (custom M/EEG)', 'Empirical (forced high-fidelity selection)'; ...
                                                'auto', 'fs_precomputed', 'fs_interpolated', 'bst_headmodel', 'empirical'};
    sProcess.options.ref_matrix_type.Type    = 'radio_label';
    sProcess.options.ref_matrix_type.Value   = 'auto';
    % === Parallel processing
    sProcess.options.label3.Comment   = '<BR>';
    sProcess.options.label3.Type      = 'label';
    sProcess.options.parallel.Comment = 'Use parallel processing (N.B. needs a lot more RAM)';
    sProcess.options.parallel.Type    = 'checkbox';
    sProcess.options.parallel.Value   = 1;
    % === Visualize artifacts
    sProcess.options.visualize_artifacts.Comment = 'Visualize artifacts';
    sProcess.options.visualize_artifacts.Type    = 'checkbox';
    sProcess.options.visualize_artifacts.Value   = 0;
    % === SENSAI visualization
    sProcess.options.visualize_sensai.Comment = 'SENSAI visualization';
    sProcess.options.visualize_sensai.Type    = 'checkbox';
    sProcess.options.visualize_sensai.Value   = 0;
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
end


%% ===== GET OPTIONS =====
function [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, visualize_sensai, enova_threshold, save_artifacts] = GetOptions(sProcess)
    artifact_threshold_type = sProcess.options.artifact_threshold_type.Value;
    epoch_size_in_cycles    = sProcess.options.epoch_size_in_cycles.Value{1};
    lowcut_frequency        = sProcess.options.lowcut_frequency.Value{1};
    switch sProcess.options.ref_matrix_type.Value
        case 'auto',            ref_matrix_type = 'Auto';
        case 'fs_precomputed',  ref_matrix_type = 'Freesurfer (precomputed)';
        case 'fs_interpolated', ref_matrix_type = 'Freesurfer (interpolated)';
        case 'bst_headmodel',   ref_matrix_type = 'Brainstorm leadfield';
        case 'empirical',       ref_matrix_type = 'Empirical';
        otherwise,              ref_matrix_type = 'Unknown';
    end
    parallel             = sProcess.options.parallel.Value;
    visualize_artifacts  = sProcess.options.visualize_artifacts.Value;
    visualize_sensai     = sProcess.options.visualize_sensai.Value;
    if isfield(sProcess.options, 'save_artifacts') && isfield(sProcess.options.save_artifacts, 'Value')
        save_artifacts = sProcess.options.save_artifacts.Value;
    else
        save_artifacts = 0;
    end
    if sProcess.options.reject_by_enova.Value
        enova_threshold = sProcess.options.enova_threshold.Value{1};
    else
        enova_threshold = [];
    end
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, ~, ~, enova_threshold, ~] = GetOptions(sProcess);
    Comment = ['GEDAI: ' artifact_threshold_type ', ' num2str(epoch_size_in_cycles) ' cycles, ' num2str(lowcut_frequency) ' Hz, ' ref_matrix_type];
    if ~isempty(enova_threshold)
        Comment = [Comment, ', ENOVA=' num2str(enova_threshold)];
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
    [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, visualize_sensai, enova_threshold, save_artifacts] = GetOptions(sProcess);

    % Iterate over inputs
    for iInput = 1:length(sInputs)
        sInput = sInputs(iInput);

        try
            % =========================================================
            % STEP 1: LOAD DATA
            % =========================================================
            % Load the BST file structure (works for both raw links and imported data)
            FileMat = in_bst_data(sInput.FileName);

            isRaw = strcmpi(FileMat.DataType, 'raw');

            if isRaw
                % For raw links, FileMat.F is the sFile descriptor struct.
                % Use in_fread with the correct Brainstorm API.
                disp(['GEDAI> Reading raw file: ' sInput.FileName]);
                sFile = FileMat.F;

                % Get channel file FIRST (required by in_fread)
                [sChannel, iStudyChannel] = bst_get('ChannelForStudy', sInput.iStudy);
                ChannelMat = in_bst_channel(sChannel.FileName);

                % Compute sample bounds from sFile.prop.times (seconds) — 0-indexed integers
                SamplesBounds = round(sFile.prop.times * sFile.prop.sfreq);

                % Read all channels, all samples, epoch 1
                [DataMatrix, TimeVector] = in_fread(sFile, ChannelMat, 1, SamplesBounds);

                sInput.A          = DataMatrix;
                sInput.TimeVector = TimeVector;
                sInput.Comment    = FileMat.Comment;
                if isfield(FileMat, 'History'),     sInput.History     = FileMat.History;     end
                if isfield(FileMat, 'Events'),      sInput.Events      = FileMat.Events;      end
                if isfield(FileMat, 'ChannelFlag'), sInput.ChannelFlag = FileMat.ChannelFlag; end

            else
                % Imported data: data matrix is directly in FileMat.F
                sInput.A          = FileMat.F;
                sInput.TimeVector = FileMat.Time;
                sInput.Comment    = FileMat.Comment;
                if isfield(FileMat, 'History'),     sInput.History     = FileMat.History;     end
                if isfield(FileMat, 'Events'),      sInput.Events      = FileMat.Events;      end
                if isfield(FileMat, 'ChannelFlag'), sInput.ChannelFlag = FileMat.ChannelFlag; end

                % Get channel file
                [sChannel, iStudyChannel] = bst_get('ChannelForStudy', sInput.iStudy);
                ChannelMat = in_bst_channel(sChannel.FileName);
            end

            % =========================================================
            % STEP 2: CHANNEL SELECTION
            % =========================================================
            eeg_meg_idx = find(ismember({ChannelMat.Channel.Type}, {'EEG', 'MEG', 'MEG MAG', 'MEG GRAD'}));
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
            is_empirical = ismember(sProcess.options.ref_matrix_type.Value, {'auto', 'empirical'});
            if strcmp(ref_matrix_type, 'Brainstorm leadfield') || is_empirical
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
                    % For "Brainstorm leadfield", a head model is MANDATORY
                    if strcmp(ref_matrix_type, 'Brainstorm leadfield')
                        bst_report('Error', sProcess, sInput, 'No head model found.');
                        continue;
                    else
                        % For "Auto" or "Empirical", a head model is OPTIONAL (will fallback to standard template)
                        disp('GEDAI> No Brainstorm head model found. Proceeding with standard template anchor.');
                    end
                else
                    % Head model found, load it
                    HeadModel    = in_bst_headmodel(HeadModelFile, 0, 'Gain');
                    Gain_filtered = HeadModel.Gain(eeg_meg_idx, :);
                    if strcmp(signal_type, 'eeg')
                        Gain_avref = Gain_filtered - mean(Gain_filtered, 1);
                    else
                        Gain_avref = Gain_filtered;
                    end
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
                if strcmp(sProcess.options.ref_matrix_type.Value, 'bst_headmodel') || (is_empirical && ~isempty(Gain_avref))
                    Gain_MAG = Gain_avref(mag_idx_in_filtered, :);
                    ref_matrix_param_MAG = Gain_MAG * Gain_MAG';
                elseif strcmp(sProcess.options.ref_matrix_type.Value, 'fs_precomputed')
                    ref_matrix_param_MAG = 'precomputed';
                elseif strcmp(sProcess.options.ref_matrix_type.Value, 'fs_interpolated')
                    ref_matrix_param_MAG = 'interpolated';
                else
                    ref_matrix_param_MAG = sProcess.options.ref_matrix_type.Value; % 'empirical'
                end
                
                if is_empirical
                    [EEGclean_MAG, EEGartifacts_MAG] = GEDAI_empirical(EEG_MAG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_MAG, parallel, visualize_artifacts, enova_threshold, signal_type, visualize_sensai);
                else
                    [EEGclean_MAG, EEGartifacts_MAG] = GEDAI(EEG_MAG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_MAG, parallel, visualize_artifacts, enova_threshold, signal_type, visualize_sensai);
                end

                % --- GRAD ---
                ChannelMatGRAD = ChannelMatFiltered;
                ChannelMatGRAD.Channel = ChannelMatFiltered.Channel(grad_idx_in_filtered);
                sInputGRAD   = sInputFiltered;
                sInputGRAD.A = sInputFiltered.A(grad_idx_in_filtered, :);
                EEG_GRAD = brainstorm2eeglab(sInputGRAD, ChannelMatGRAD);
                if length(sInputGRAD.TimeVector) > 1, EEG_GRAD.srate = 1 / mean(diff(sInputGRAD.TimeVector)); end
                if strcmp(sProcess.options.ref_matrix_type.Value, 'bst_headmodel') || (is_empirical && ~isempty(Gain_avref))
                    Gain_GRAD = Gain_avref(grad_idx_in_filtered, :);
                    ref_matrix_param_GRAD = Gain_GRAD * Gain_GRAD';
                elseif strcmp(sProcess.options.ref_matrix_type.Value, 'fs_precomputed')
                    ref_matrix_param_GRAD = 'precomputed';
                elseif strcmp(sProcess.options.ref_matrix_type.Value, 'fs_interpolated')
                    ref_matrix_param_GRAD = 'interpolated';
                else
                    ref_matrix_param_GRAD = sProcess.options.ref_matrix_type.Value; % 'empirical'
                end
                
                if is_empirical
                    [EEGclean_GRAD, EEGartifacts_GRAD] = GEDAI_empirical(EEG_GRAD, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_GRAD, parallel, visualize_artifacts, enova_threshold, signal_type, visualize_sensai);
                else
                    [EEGclean_GRAD, EEGartifacts_GRAD] = GEDAI(EEG_GRAD, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_GRAD, parallel, visualize_artifacts, enova_threshold, signal_type, visualize_sensai);
                end

                % --- Recombine ---
                EEGclean = brainstorm2eeglab(sInputFiltered, ChannelMatFiltered);
                EEGclean.data(mag_idx_in_filtered, :)  = EEGclean_MAG.data;
                EEGclean.data(grad_idx_in_filtered, :) = EEGclean_GRAD.data;
                EEGartifacts = EEGclean;
                EEGartifacts.data(mag_idx_in_filtered, :)  = EEGartifacts_MAG.data;
                EEGartifacts.data(grad_idx_in_filtered, :) = EEGartifacts_GRAD.data;

                % Combine rejection masks
                if isfield(EEGclean_MAG.etc, 'GEDAI') && isfield(EEGclean_MAG.etc.GEDAI, 'samples_to_keep') && ...
                   isfield(EEGclean_GRAD.etc, 'GEDAI') && isfield(EEGclean_GRAD.etc.GEDAI, 'samples_to_keep')
                    mask_MAG  = EEGclean_MAG.etc.GEDAI.samples_to_keep;
                    mask_GRAD = EEGclean_GRAD.etc.GEDAI.samples_to_keep;
                    if length(mask_MAG) == length(mask_GRAD)
                        combined_mask = mask_MAG & mask_GRAD;
                        EEGclean.etc.GEDAI.samples_to_keep    = combined_mask;
                        EEGclean.etc.GEDAI.percentage_rejected = 100 * sum(~combined_mask) / length(combined_mask);
                    else
                        EEGclean.etc.GEDAI = EEGclean_MAG.etc.GEDAI;
                    end
                elseif isfield(EEGclean_MAG.etc, 'GEDAI')
                    EEGclean.etc.GEDAI = EEGclean_MAG.etc.GEDAI;
                elseif isfield(EEGclean_GRAD.etc, 'GEDAI')
                    EEGclean.etc.GEDAI = EEGclean_GRAD.etc.GEDAI;
                end

            else
                % Single processing (EEG or generic MEG)
                sInputFiltered   = sInput;
                sInputFiltered.A = sInput.A(eeg_meg_idx, :);
                EEG = brainstorm2eeglab(sInputFiltered, ChannelMatFiltered);
                if length(sInputFiltered.TimeVector) > 1
                    EEG.srate = 1 / mean(diff(sInputFiltered.TimeVector));
                end
                if strcmp(sProcess.options.ref_matrix_type.Value, 'bst_headmodel') || (is_empirical && ~isempty(Gain_avref))
                    ref_matrix_param = Gain_avref * Gain_avref';
                elseif strcmp(sProcess.options.ref_matrix_type.Value, 'fs_precomputed')
                    ref_matrix_param = 'precomputed';
                elseif strcmp(sProcess.options.ref_matrix_type.Value, 'fs_interpolated')
                    ref_matrix_param = 'interpolated';
                else
                    ref_matrix_param = sProcess.options.ref_matrix_type.Value; % 'empirical'
                end
                
                if is_empirical
                    [EEGclean, EEGartifacts] = GEDAI_empirical(EEG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param, parallel, visualize_artifacts, enova_threshold, signal_type, visualize_sensai);
                else
                    [EEGclean, EEGartifacts] = GEDAI(EEG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param, parallel, visualize_artifacts, enova_threshold, signal_type, visualize_sensai);
                end
                
            end

            % =========================================================
            % STEP 5: BUILD OUTPUT STRUCTURES
            % =========================================================
            % Compute comment strings FIRST (used by both Cleaned and Artifacts)
            current_comment = sInput.Comment;
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
            if isfield(sInput, 'ChannelFlag') && length(sInput.ChannelFlag) == nChannelsTotal
                ChannelFlagOut = sInput.ChannelFlag(:);
            else
                ChannelFlagOut = ones(nChannelsTotal, 1);
            end

            % --- Cleaned file ---
            FileMatCleaned.Comment     = ['Cleaned | ', current_comment, ' | ', gedai_params];
            FileMatCleaned.DataType    = 'recordings';
            FileMatCleaned.Time        = TimeOut;
            FileMatCleaned.F           = DataOut;
            FileMatCleaned.ChannelFlag = ChannelFlagOut;
            if isfield(sInput, 'Events'),  FileMatCleaned.Events  = sInput.Events;  end
            if isfield(sInput, 'History'), FileMatCleaned.History = sInput.History; end

            % IMPORTANT: Save to the study folder, NOT the raw subfolder.
            % Brainstorm detects raw files by checking if the filename contains 'data_0raw'.
            % If we save inside the @raw... subfolder, the path will contain 'data_0raw'
            % and Brainstorm will treat our imported file as a raw link, causing crashes.
            sStudyOut   = bst_get('Study', sInput.iStudy);
            StudyFolder = bst_fileparts(file_fullpath(sStudyOut.FileName));
            CleanedFileName = bst_process('GetNewFilename', StudyFolder, 'data_gedai_cleaned');
            bst_save(CleanedFileName, FileMatCleaned, 'v6');
            db_add_data(sInput.iStudy, CleanedFileName, FileMatCleaned);
            OutputFiles{end+1} = CleanedFileName;


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

                    FileMatArtifacts.Comment     = ['Artifacts | ', current_comment, ' | ', gedai_params];
                    FileMatArtifacts.DataType    = 'recordings';
                    FileMatArtifacts.Time        = TimeOut;
                    FileMatArtifacts.F           = ArtifactData;
                    FileMatArtifacts.ChannelFlag = ChannelFlagOut;
                    if isfield(sInput, 'Events'),  FileMatArtifacts.Events  = sInput.Events;  end
                    if isfield(sInput, 'History'), FileMatArtifacts.History = sInput.History; end

                    ArtifactsFileName = bst_process('GetNewFilename', StudyFolder, 'data_gedai_artifacts');
                    bst_save(ArtifactsFileName, FileMatArtifacts, 'v6');
                    db_add_data(sInput.iStudy, ArtifactsFileName, FileMatArtifacts);
                    OutputFiles{end+1} = ArtifactsFileName;

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
end

function sOutput = eeglab2brainstorm(EEG, sInput)
    sOutput = sInput;
    sOutput.A = EEG.data / 1e6; % µV -> V
    if length(sOutput.TimeVector) ~= size(EEG.data, 2)
        sOutput.TimeVector = EEG.times / 1000;
    end
    sOutput.Comment = EEG.setname;
end
