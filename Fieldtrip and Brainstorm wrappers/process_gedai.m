function varargout = process_gedai( varargin )
% PROCESS_GEDAI: Wrapper for GEDAI.m function to be used in Brainstorm
%
% USAGE:                sProcess = process_gedai('GetDescription')
%                         sInput = process_gedai('Run', sProcess, sInput)

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
%
% Authors: Tomas Ros, Center for Biomedical Imaging (CIBM), University of Geneva, 2025

eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    sProcess.Comment     = 'GEDAI';
    sProcess.FileTag     = 'gedai';
    sProcess.Category    = 'Filter';
    sProcess.SubGroup    = 'Artifacts';
    sProcess.Index       = 113.7;
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data', 'raw'};
    sProcess.OutputTypes = {'data', 'raw'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    sProcess.Description = 'https://neuroimage.usc.edu/brainstorm/Tutorials/Gedai';
    sProcess.isSeparator = 1;
    % Definition of the options
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
    sProcess.options.ref_matrix_type.Comment = {'Freesurfer (precomputed for EEG)', 'Freesurfer (interpolated for EEG)', 'Brainstorm headmodel (custom for M/EEG)'; ...
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
    sProcess.options.visualize_artifacts.Value   = 0;
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
    % === Use modwtdetails
    sProcess.options.label5.Comment = '<BR>';
    sProcess.options.label5.Type    = 'label';
    sProcess.options.use_modwtdetails.Comment = 'Use wavelet packet transform (modwpt)';
    sProcess.options.use_modwtdetails.Type    = 'checkbox';
    sProcess.options.use_modwtdetails.Value   = 0;
end


%% ===== GET OPTIONS =====
function [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, enova_threshold, use_modwpt] = GetOptions(sProcess)
    % Artifact threshold type
    artifact_threshold_type = sProcess.options.artifact_threshold_type.Value;
    % Epoch size in cycles
    epoch_size_in_cycles = sProcess.options.epoch_size_in_cycles.Value{1};
    % Low-cut frequency
    lowcut_frequency = sProcess.options.lowcut_frequency.Value{1};
    % Reference matrix type
    switch sProcess.options.ref_matrix_type.Value
        case 'fs_precomputed'
            ref_matrix_type = 'Freesurfer (precomputed)';
        case 'fs_interpolated'
            ref_matrix_type = 'Freesurfer (interpolated)';
        case 'bst_headmodel'
            ref_matrix_type = 'Brainstorm leadfield';
    end
    % Parallel processing
    parallel = sProcess.options.parallel.Value;
    % Visualize artifacts
    visualize_artifacts = sProcess.options.visualize_artifacts.Value;
    % ENOVA bad epoch rejection
    if sProcess.options.reject_by_enova.Value
        enova_threshold = sProcess.options.enova_threshold.Value{1};
    else
        enova_threshold = [];
    end
    % Use modwtdetails
    use_modwpt = sProcess.options.use_modwtdetails.Value;
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, ~, ~, enova_threshold, use_modwpt] = GetOptions(sProcess);
    
    if use_modwpt
        wavelet_desc = 'modwpt';
    else
        wavelet_desc = 'modwt';
    end
    
    Comment = ['GEDAI: ' wavelet_desc ', ' artifact_threshold_type ', ' num2str(epoch_size_in_cycles) ' cycles, ' num2str(lowcut_frequency) ' Hz, ' ref_matrix_type];
    if ~isempty(enova_threshold)
        Comment = [Comment, ', ENOVA=' num2str(enova_threshold)];
    end
end


%% ===== RUN =====
function sInput = Run(sProcess, sInput) %#ok<DEFNU>
    % Check if GEDAI plugin is loaded
    unloadPlug = 0;
    PlugDesc = bst_plugin('GetDescription', 'gedai');
    if ~isequal(PlugDesc.isLoaded, 1) || isempty(PlugDesc.Path)
        [isOk, errMsg] = bst_plugin('Load', 'gedai');
        if ~isOk
            bst_report('Error', sProcess, sInput, errMsg);
            return
        end
        unloadPlug = 1;
    end
    % Get options
    [artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, enova_threshold, use_modwpt] = GetOptions(sProcess);
    
    fprintf('GEDAI> Running with use_modwpt = %d\n', use_modwpt);

    % Get channel file for study
    [sChannel, iStudyChannel] = bst_get('ChannelForStudy', sInput.iStudy);
    % Load channel file
    ChannelMat = in_bst_channel(sChannel.FileName);

    % Filter for EEG and MEG channels only
    eeg_meg_idx = find(ismember({ChannelMat.Channel.Type}, {'EEG', 'MEG', 'MEG MAG', 'MEG GRAD'}));
    if isempty(eeg_meg_idx)
        bst_report('Error', sProcess, sInput, 'No EEG or MEG channels found in the data.');
        return;
    end
    
    % Create filtered channel structure
    ChannelMatFiltered = ChannelMat;
    ChannelMatFiltered.Channel = ChannelMat.Channel(eeg_meg_idx);
    
    % Detect signal type from channel labels
    channel_types = {ChannelMatFiltered.Channel.Type};
    eeg_count = sum(strcmp(channel_types, 'EEG'));
    meg_count = sum(ismember(channel_types, {'MEG', 'MEG MAG', 'MEG GRAD'}));
    
    % Validate that channels are not mixed
    if eeg_count > 0 && meg_count > 0
        bst_report('Error', sProcess, sInput, 'Cannot process mixed EEG and MEG channels. They require different leadfield models. Please process them separately.');
        return;
    elseif eeg_count > 0
        signal_type = 'eeg';
        process_mag_grad_separately = false;
    elseif meg_count > 0
        signal_type = 'meg';
        % For MEG, detect if we have MAG and/or GRAD channels
        mag_count = sum(strcmp(channel_types, 'MEG MAG'));
        grad_count = sum(strcmp(channel_types, 'MEG GRAD'));
        meg_generic_count = sum(strcmp(channel_types, 'MEG'));
        
        % Debug output - show unique channel types
        unique_types = unique(channel_types);
        fprintf('GEDAI> Unique channel types found: %s\\n', strjoin(unique_types, ', '));
        fprintf('GEDAI> Channel type counts: %d MEG MAG, %d MEG GRAD, %d generic MEG (Total: %d)\\n', mag_count, grad_count, meg_generic_count, meg_count);
        
        process_mag_grad_separately = (mag_count > 0 && grad_count > 0);
    else
        bst_report('Error', sProcess, sInput, 'No valid EEG or MEG channels detected.');
        return;
    end
    
    % Handle ref_matrix_type and prepare Gain matrix
    if strcmp(ref_matrix_type, 'Brainstorm leadfield')
        HeadModelFile = [];
        
        % Get Data Study and Channel Study
        sStudyData = bst_get('Study', sInput.iStudy);
        sStudyChan = bst_get('Study', iStudyChannel);

        % Strategy 1: Active/Default HeadModel in Data Study
        if ~isempty(sStudyData.iHeadModel) && ~isempty(sStudyData.HeadModel)
             HeadModelFile = sStudyData.HeadModel(sStudyData.iHeadModel).FileName;
        end
        
        % Strategy 2: Active/Default HeadModel in Channel Study
        if isempty(HeadModelFile) && ~isempty(sStudyChan.iHeadModel) && ~isempty(sStudyChan.HeadModel)
             HeadModelFile = sStudyChan.HeadModel(sStudyChan.iHeadModel).FileName;
        end
        
        % Strategy 3: Any HeadModel in Data Study (First available)
        if isempty(HeadModelFile) && ~isempty(sStudyData.HeadModel)
            HeadModelFile = sStudyData.HeadModel(1).FileName;
        end
        
        % Strategy 4: Any HeadModel in Channel Study (First available)
        if isempty(HeadModelFile) && ~isempty(sStudyChan.HeadModel)
            HeadModelFile = sStudyChan.HeadModel(1).FileName;
        end
        
        if isempty(HeadModelFile)
             bst_report('Error', sProcess, sInput, 'No head model found in Data or Channel studies (checked for defaults and any available models).');
             return;
        end
        
        HeadModel = in_bst_headmodel(HeadModelFile, 0, 'Gain');
        
        % Filter Gain matrix to match EEG/MEG channels only
        Gain_filtered = HeadModel.Gain(eeg_meg_idx, :);
        
        % Apply average reference to leadfield (only for EEG)
        if strcmp(signal_type, 'eeg')
            Gain_avref = Gain_filtered - mean(Gain_filtered, 1);
        else
            % For MEG, skip average referencing
            Gain_avref = Gain_filtered;
        end
    end

    % Process channels based on type
    if process_mag_grad_separately
        % Process MAG and GRAD separately for MEG data
        fprintf('GEDAI> Processing MAG and GRAD channels separately (%d MAG, %d GRAD channels)\n', mag_count, grad_count);
        
        % Identify MAG and GRAD channel indices within the filtered channel set
        mag_idx_in_filtered = find(strcmp(channel_types, 'MEG MAG'));
        grad_idx_in_filtered = find(strcmp(channel_types, 'MEG GRAD'));
        
        % Create filtered input structure with only EEG/MEG channel data
        sInputFiltered = sInput;
        sInputFiltered.A = sInput.A(eeg_meg_idx, :);
        
        % ===== Process MAG channels =====
        fprintf('GEDAI> Processing %d MAG channels...\n', length(mag_idx_in_filtered));
        ChannelMatMAG = ChannelMatFiltered;
        ChannelMatMAG.Channel = ChannelMatFiltered.Channel(mag_idx_in_filtered);
        
        sInputMAG = sInputFiltered;
        sInputMAG.A = sInputFiltered.A(mag_idx_in_filtered, :);
        
        EEG_MAG = brainstorm2eeglab(sInputMAG, ChannelMatMAG);
        if length(sInputMAG.TimeVector) > 1
            EEG_MAG.srate = 1 / mean(diff(sInputMAG.TimeVector));
        end
        
        % Compute MAG-specific reference covariance matrix
        if strcmp(ref_matrix_type, 'Brainstorm leadfield')
            Gain_MAG = Gain_avref(mag_idx_in_filtered, :);
            ref_matrix_param_MAG = Gain_MAG * Gain_MAG';
        elseif strcmp(ref_matrix_type, 'Freesurfer (precomputed)')
            ref_matrix_param_MAG = 'precomputed';
        elseif strcmp(ref_matrix_type, 'Freesurfer (interpolated)')
            ref_matrix_param_MAG = 'interpolated';
        end
        
        % Run GEDAI for MAG channels
        EEGclean_MAG = GEDAI(EEG_MAG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_MAG, parallel, visualize_artifacts, enova_threshold, signal_type, use_modwpt);
        
        % ===== Process GRAD channels =====
        fprintf('GEDAI> Processing %d GRAD channels...\n', length(grad_idx_in_filtered));
        ChannelMatGRAD = ChannelMatFiltered;
        ChannelMatGRAD.Channel = ChannelMatFiltered.Channel(grad_idx_in_filtered);
        
        sInputGRAD = sInputFiltered;
        sInputGRAD.A = sInputFiltered.A(grad_idx_in_filtered, :);
        
        EEG_GRAD = brainstorm2eeglab(sInputGRAD, ChannelMatGRAD);
        if length(sInputGRAD.TimeVector) > 1
            EEG_GRAD.srate = 1 / mean(diff(sInputGRAD.TimeVector));
        end
        
        % Compute GRAD-specific reference covariance matrix
        if strcmp(ref_matrix_type, 'Brainstorm leadfield')
            Gain_GRAD = Gain_avref(grad_idx_in_filtered, :);
            ref_matrix_param_GRAD = Gain_GRAD * Gain_GRAD';
        elseif strcmp(ref_matrix_type, 'Freesurfer (precomputed)')
            ref_matrix_param_GRAD = 'precomputed';
        elseif strcmp(ref_matrix_type, 'Freesurfer (interpolated)')
            ref_matrix_param_GRAD = 'interpolated';
        end
        
        % Run GEDAI for GRAD channels
        EEGclean_GRAD = GEDAI(EEG_GRAD, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param_GRAD, parallel, visualize_artifacts, enova_threshold, signal_type, use_modwpt);
        
        % ===== Combine MAG and GRAD results =====
        % Create combined EEG structure
        EEGclean = brainstorm2eeglab(sInputFiltered, ChannelMatFiltered);
        
        % Reconstruct data in original channel order
        EEGclean.data(mag_idx_in_filtered, :) = EEGclean_MAG.data;
        EEGclean.data(grad_idx_in_filtered, :) = EEGclean_GRAD.data;
        
        % Handle epoch rejection: combine masks from both MAG and GRAD
        if isfield(EEGclean_MAG.etc, 'GEDAI') && isfield(EEGclean_MAG.etc.GEDAI, 'samples_to_keep') && ...
           isfield(EEGclean_GRAD.etc, 'GEDAI') && isfield(EEGclean_GRAD.etc.GEDAI, 'samples_to_keep')
            % Use intersection of both masks (reject if either MAG or GRAD rejected)
            mask_MAG = EEGclean_MAG.etc.GEDAI.samples_to_keep;
            mask_GRAD = EEGclean_GRAD.etc.GEDAI.samples_to_keep;
            
            % Ensure masks are the same length
            if length(mask_MAG) == length(mask_GRAD)
                combined_mask = mask_MAG & mask_GRAD;
                EEGclean.etc.GEDAI.samples_to_keep = combined_mask;
                
                % Calculate combined rejection percentage
                original_samples = length(combined_mask);
                kept_samples = sum(combined_mask);
                EEGclean.etc.GEDAI.percentage_rejected = 100 * (1 - kept_samples / original_samples);
            else
                % If masks differ in length, use MAG mask as default
                EEGclean.etc.GEDAI = EEGclean_MAG.etc.GEDAI;
            end
        elseif isfield(EEGclean_MAG.etc, 'GEDAI')
            EEGclean.etc.GEDAI = EEGclean_MAG.etc.GEDAI;
        elseif isfield(EEGclean_GRAD.etc, 'GEDAI')
            EEGclean.etc.GEDAI = EEGclean_GRAD.etc.GEDAI;
        end
        
    else
        % Process all channels together (EEG or single MEG type)
        
        % Create filtered input structure with only EEG/MEG channel data
        sInputFiltered = sInput;
        sInputFiltered.A = sInput.A(eeg_meg_idx, :);

        % Convert Brainstorm sInput to EEGLAB format
        EEG = brainstorm2eeglab(sInputFiltered, ChannelMatFiltered);

        % Explicitly ensure sampling rate is correct based on the input TimeVector
        if length(sInputFiltered.TimeVector) > 1
            EEG.srate = 1 / mean(diff(sInputFiltered.TimeVector));
        end

        % Compute reference covariance matrix
        if strcmp(ref_matrix_type, 'Brainstorm leadfield')
            ref_matrix_param = Gain_avref * Gain_avref';
        elseif strcmp(ref_matrix_type, 'Freesurfer (precomputed)')
            ref_matrix_param = 'precomputed';
        elseif strcmp(ref_matrix_type, 'Freesurfer (interpolated)')
            ref_matrix_param = 'interpolated';
        end

        % Run GEDAI
        EEGclean = GEDAI(EEG, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_param, parallel, visualize_artifacts, enova_threshold, signal_type, use_modwpt);
    end
    
    % Map cleaned EEG/MEG data back to original channel positions
    sOutput = sInput;
    
    % Check if epochs were rejected and apply mask to non-EEG channels
    if isfield(EEGclean.etc, 'GEDAI') && isfield(EEGclean.etc.GEDAI, 'samples_to_keep')
        mask = EEGclean.etc.GEDAI.samples_to_keep;
        if length(mask) == size(sOutput.A, 2)
            sOutput.A = sOutput.A(:, mask);
            if length(sOutput.TimeVector) == length(mask)
                 sOutput.TimeVector = sOutput.TimeVector(mask);
            end
        end
    end

    sOutput.A(eeg_meg_idx, :) = EEGclean.data;  % Replace only EEG/MEG channels with cleaned data
    
    % Convert back to Brainstorm format
    sInput = eeglab2brainstorm(EEGclean, sInputFiltered);
    sInput.A = sOutput.A;  % Use the full channel data with cleaned EEG
    
    % Update Comment logic
    new_duration = sInput.TimeVector(end) - sInput.TimeVector(1);
    
    % Get the base comment
    current_comment = sInput.Comment;
    
    % Strip old duration tag
    current_comment = regexprep(current_comment, ' \([\d\.]+s,[\d\.]+s\)', ''); 
    
    % Generate GEDAI parameters tag
    gedai_params = FormatComment(sProcess);
    
    % Append stats
    if isfield(EEGclean.etc, 'GEDAI')
        rej_percent = EEGclean.etc.GEDAI.percentage_rejected;
        gedai_params = [gedai_params, sprintf(', Rej=%.1f%% (%.1fs)', rej_percent, new_duration)];
    end
    
    % Explicitly set the full Comment string to ensure it appears
    sInput.Comment = [current_comment, ' | ', gedai_params];
    
    % Clear CommentTag to avoid duplication if Brainstorm tries to auto-append
    sInput.CommentTag = []; 

    if isfield(sInput, 'Std') && ~isempty(sInput.Std)
        sInput.Std = [];
    end
    % Unload GEDAI plugin if loaded by this process
    % if unloadPlug
    %     bst_plugin('Unload', 'gedai');
    % end
end

%% ===== HELPER FUNCTIONS =====
function EEG = brainstorm2eeglab(sInput, ChannelMat)
    % Create an EEGLAB EEG structure populated with fields from sInput
    EEG.setname = sInput.Comment;
    EEG.filename = sInput.FileName;
    EEG.filepath = fileparts(sInput.FileName);
    EEG.subject = '';
    EEG.group = '';
    EEG.condition = '';
    EEG.session = [];
    EEG.nbchan = size(sInput.A, 1);
    EEG.trials = 1;
    EEG.pnts = size(sInput.A, 2);
    EEG.srate = 1 / (sInput.TimeVector(2) - sInput.TimeVector(1));
    EEG.xmin = sInput.TimeVector(1);
    EEG.xmax = sInput.TimeVector(end);
    EEG.times = sInput.TimeVector * 1000; % Convert to ms
    EEG.data = sInput.A * 1e6;            % Convert to uV
    EEG.etc = [];
    EEG.event = [];

    % Populate chanlocs
    for i = 1:length(ChannelMat.Channel)
        EEG.chanlocs(i).labels = ChannelMat.Channel(i).Name;
        if ~isempty(ChannelMat.Channel(i).Loc)
            EEG.chanlocs(i).X = ChannelMat.Channel(i).Loc(1) * 1000; % Convert to mm
            EEG.chanlocs(i).Y = ChannelMat.Channel(i).Loc(2) * 1000; % Convert to mm
            EEG.chanlocs(i).Z = ChannelMat.Channel(i).Loc(3) * 1000; % Convert to mm
        end
        EEG.chanlocs(i).type = ChannelMat.Channel(i).Type;
    end
end

function sOutput = eeglab2brainstorm(EEG, sInput)
    % Create a copy of the input structure
    sOutput = sInput;

    % Update the data
    sOutput.A = EEG.data;

    % Update the time vector if it has changed
    if (length(sOutput.TimeVector) ~= size(EEG.data, 2))
        sOutput.TimeVector = EEG.times / 1000; % Convert back to seconds
    end

    % Update the comment
    sOutput.Comment = EEG.setname;
end
