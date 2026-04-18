%   GEDAI() - This is the main function used to denoise EEG data using
%   generalized eigenvalue decomposition coupled with an EEG leadfield matrix
%
%   GEDAI (Generalized Eigenvalue Deartifacting Instrument)
%
% Usage:
% Example 1: Using all default values (and no bad epoch rejection)
%    >>  [EEG] = GEDAI(EEG);
%
% Example 2: Defining some parameters
%    >>  [EEG] = GEDAI(EEG, 'auto', 12, 0.5, 'precomputed', true, false, 0.9);
%
% Example 3: Using a "custom" [channel x channel] reference matrix 
%    >>  [EEG] = GEDAI(EEG, 'auto', 12, 0.5, your_refCOV);
%
% Inputs: 
% 
%   EEGin                       - EEG data in EEGlab format
% 
%   artifact_threshold_type     - Variable determining deartifacting
%                                 strength. Stronger threshold type
%                                 ("auto+") might remove more noise at the
%                                 expense of signal, while milder threshold
%                                 ("auto-") might retain more signal at the 
%                                 expense of noise. Possible levels:
%                                 "auto-", "auto" or "auto+". 
%                                 Default is "auto".
%                             
%   epoch_size_in_cycles        - Epoch size in number of wave cycles for each
%                                 wavelet band. Default is 12.
%
%   lowcut_frequency            - Low-cut frequency in Hz. Wavelet bands below this
%                                 frequency will be excluded. Default is 0.5 Hz.
% 
%   ref_matrix_type             - Matrix used as a reference for deartifacting.
%
%                                  The default "precomputed" uses a BEM leadfield for
%                                  standard electrode locations precomputed through 
%                                  OPENMEEG (343 electrodes) based on 10-5 system. 
%
%                                 "interpolated" uses the precomputed leadfield and 
%                                 interpolates it to non-standard electrode locations.
%
%                                 Altenatively, you can input a "custom" covariance matrix
%                                 (with dimensions channel x channel) via a matlab variable
% 
% 
%   parallel                    - Boolean for using parallel ('multicore') processing 
% 
%   visualize_artifacts         - Boolean for artifact visualization 
%                                 using vis_artifacts function from the ASR toolbox
%
%   ENOVA_threshold             - Threshold for rejecting epochs based on Explained
%                                 Noise Variance (ENOVA). Epochs with ENOVA >
%                                 ENOVA_threshold will be removed. Default is inf
%                                 (no rejection).
%
%   signal_type                 - Type of signal: 'eeg' or 'meg'. Default is 'eeg'.
%                                 For EEG, average referencing is applied.
%                                 For MEG, average referencing is skipped.
%    
% Outputs:
% 
%   EEGclean                - Cleaned EEG data in EEGLab struct format
% 
%   EEGartifacts            - EEG data containing only the removed artifacts
%                             (i.e. noise that was removed from EEGin)
%                             EEGin.data = EEGclean.data + EEGartifacts.data
% 
%   SENSAI_score            - Relative denoising quality score (%)
%
%   SENSAI_score_per_band   - Relative denoising quality score per band (%)
% 
%   artifact_threshold_per_band  - Vector of artifact thresholds used for each 
%                                  frequency band, starting with the broadband
%                                  approx: [broadband gamma beta alpha theta delta etc.]
%
%   mean_ENOVA              - Mean Explained Noise Variance (ENOVA) across all epochs.
%                             ENOVA is the variance of the removed noise, expressed as a 
%                             proportion of the variance of the original EEG data.
%
%   ENOVA_per_epoch         - Vector of ENOVA values for each epoch.
% 
%   com                     - output logging to EEG.history

% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI) v 1.5]
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

function [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band]=GEDAI(EEGin, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, ENOVA_threshold, signal_type, visualize_manifold)

if nargin < 2 || isempty(artifact_threshold_type)
    artifact_threshold_type = 'auto';
end
if nargin < 3 || isempty(epoch_size_in_cycles)
    epoch_size_in_cycles = 12;  % Note: Number of wave CYCLES per epoch across wavelet bands (default = 12 cycles)
end
if nargin < 4 || isempty(lowcut_frequency)
    lowcut_frequency = 0.5; %  exclude all wavelet bands below this frequency (default = 0.5 Hz)
end
if nargin < 5 || isempty(ref_matrix_type)
    ref_matrix_type = 'precomputed';
end
if nargin < 6 || isempty(parallel)
    parallel = true;
end
if nargin < 7 || isempty(visualize_artifacts)
    visualize_artifacts = false;
end
if nargin < 8 || isempty(ENOVA_threshold)
    ENOVA_threshold = inf; % If empty, set to infinity to disable rejection
end
if nargin < 9 || isempty(signal_type)
    signal_type = 'eeg';
end
if nargin < 10 || isempty(visualize_manifold)
    visualize_manifold = false;
end
% Validate signal_type
if ~ismember(lower(signal_type), {'eeg', 'meg'})
    error('signal_type must be either ''eeg'' or ''meg''');
end
signal_type = lower(signal_type);

p = fileparts(which('GEDAI'));
addpath(fullfile(p, 'auxiliaries'));
tStart = tic;

% Display signal type being processed
channel_type=EEGin.chanlocs(1).type;
if strcmp(signal_type, 'eeg')
    disp([newline 'GEDAI denoising of ' channel_type ' : '  num2str(size(EEGin.data,1)) ' channels']);
elseif strcmp(signal_type, 'meg')
    disp([newline 'GEDAI denoising of '  channel_type ' : ' num2str(size(EEGin.data,1)) ' channels']);
end  
% -- Ensure epoch size results in an even number of samples (for broadband)
 broadband_epoch_size = 1; % Note: IN SECONDS (this is now only the DEFAULT for broadband)
if rem(broadband_epoch_size*EEGin.srate, 2) ~= 0
    ideal_total_samples_double = broadband_epoch_size * EEGin.srate;
    nearest_integer_samples = round(ideal_total_samples_double);
    if rem(nearest_integer_samples, 2) ~= 0
        if abs(ideal_total_samples_double - (nearest_integer_samples - 1)) < abs(ideal_total_samples_double - (nearest_integer_samples + 1))
            target_total_samples_int = nearest_integer_samples - 1;
        else
            target_total_samples_int = nearest_integer_samples + 1;
        end
    else
        target_total_samples_int = nearest_integer_samples;
    end
    broadband_epoch_size = target_total_samples_int / EEGin.srate;
end

%% Ensure double input (initially)
EEGin.data=double(EEGin.data);

%% Pre-processing
if strcmp(signal_type, 'eeg')
    EEGavRef = GEDAI_nonRankDeficientAveRef(EEGin); % non rank-deficient average referencing (Makoto's plugin)
else
    % For MEG, skip average referencing
    EEGavRef = EEGin;
end

%% Create Reference Covariance Matrix (refCOV)
full_leadfield_matrix = []; % Initialize empty leadfield
if ~ischar(ref_matrix_type)
    refCOV = ref_matrix_type; % Use custom covariance matrix
    disp([newline 'Using custom covariance matrix']);
    % 
    % if strcmp(signal_type, 'meg')
    %  disp([newline 'Normalizing MEG gram matrix']);
    % refCOV=corrcov(refCOV);
    % end

 
else
    switch ref_matrix_type
        case 'precomputed'
        disp([newline 'GEDAI Leadfield model: BEM precomputed for EEG'])
            L=load('fsavLEADFIELD_4_GEDAI.mat');
            electrodes_labels = {EEGin.chanlocs.labels};
            template_electrode_labels = {L.leadfield4GEDAI.electrodes.Name};
            
            % Extract matching substrings from EEG labels
            chanidx = zeros(1, length(electrodes_labels));
            for i = 1:length(electrodes_labels)
                eeg_label = electrodes_labels{i};
                % Try direct match first
                [found, idx] = ismember(lower(eeg_label), lower(template_electrode_labels));
                if found
                    chanidx(i) = idx;
                else
                    % Search for template labels within the EEG label
                    for j = 1:length(template_electrode_labels)
                        template_label = template_electrode_labels{j};
                        % Case-insensitive substring search
                        if contains(lower(eeg_label), lower(template_label))
                            chanidx(i) = j;
                            break;
                        end
                    end
                end
            end
            
            if any(chanidx == 0)
                error('Electrode labels not found. Select "interpolated" leadfield matrix for non-standard locations.');
            end
            refCOV = L.leadfield4GEDAI.gram_matrix_avref(chanidx,chanidx);

        case 'interpolated'
    % 1. Verification of Spatial Locations
    % We check if the number of populated X and sph_theta coordinates 
    % matches the actual number of channels.
    num_chans = length(EEGavRef.chanlocs);
    has_cartesian = length([EEGavRef.chanlocs.X]) == num_chans;
    has_spherical = length([EEGavRef.chanlocs.sph_theta]) == num_chans;
    
    if ~has_cartesian || ~has_spherical
        error(['CRITICAL: Channel locations are incomplete. ' ...
               'Ensure all %d channels have X, Y, Z and spherical coordinates.'], num_chans);
    
    else
        % 2. Leadfield Processing
        disp([newline 'GEDAI Leadfield model: BEM interpolated for EEG'])
        L = load('fsavLEADFIELD_4_GEDAI.mat');
        
        % The leadfield data needs to be average referenced before interpolation
        leadfield_EEG = L.leadfield4GEDAI.EEG;
        
        % Average reference the Gain matrix (channels x sources)
        leadfield_EEG.data = L.leadfield4GEDAI.Gain - mean(L.leadfield4GEDAI.Gain, 1); 
        
        % 3. Interpolation and Covariance
        interpolated_EEG = interp_mont_GEDAI(leadfield_EEG, EEGavRef.chanlocs);
        refCOV = interpolated_EEG.data * interpolated_EEG.data';
        
        end
    end
end


% --- Wavelet-based High-Pass Filtering ---
% Calculate required level to resolve lowcut_frequency
highpass_frequency=0.1;
hp_wavelet_levels = ceil(log2(EEGavRef.srate / highpass_frequency) - 1);
% Limit to maximum possible level given data length
max_possible_level = floor(log2(size(EEGavRef.data, 2)));
hp_wavelet_levels = min(hp_wavelet_levels, max_possible_level);
% Ensure reasonable minimum
hp_wavelet_levels = max(hp_wavelet_levels, 3);
wavelet_type = 'haar';

% Decompose the signal
% Robust execution order: GPU(Double) -> GPU(Single) -> CPU(Double) -> CPU(Single)
success = false;

% disp([newline 'Wavelet high-pass filtering > ' num2str(highpass_frequency) 'Hz']);
warning('off');
% Attempt GPU Processing
if gpuDeviceCount > 0
    try
        disp('Attempting GPU processing (Double Precision)...');
        parallel.gpu.enableCUDAForwardCompatibility(true)
        data_gpu = gpuArray(EEGavRef.data');
        wpt_hp = modwt_custom(data_gpu, wavelet_type, hp_wavelet_levels);
        mra_hp = gather(modwtmra_custom(wpt_hp, wavelet_type)); 
        clear data_gpu wpt_hp;
        success = true;
    catch 
        warning('GPU (Double) failed: %s. Attempting GPU (Single Precision)...');
        try
            data_gpu = gpuArray(single(EEGavRef.data'));
            wpt_hp = modwt_custom(data_gpu, wavelet_type, hp_wavelet_levels);
            mra_hp = gather(modwtmra_custom(wpt_hp, wavelet_type)); 
            clear data_gpu wpt_hp;
            success = true;
        catch 
            warning('GPU (Single) failed: %s. Falling back to CPU.');
        end
    end
end

% Fallback to CPU if GPU failed or unavailable
if ~success
    try
        disp('Attempting CPU processing (Double Precision)...');
        wpt_hp = modwt_custom(EEGavRef.data', wavelet_type, hp_wavelet_levels);
        mra_hp = modwtmra_custom(wpt_hp, wavelet_type);
        clear wpt_hp;
    catch 
        warning('CPU (Double) failed: %s. Attempting CPU (Single Precision)...');
        % Single precision fallback for OOM
        wpt_hp = modwt_custom(single(EEGavRef.data'), wavelet_type, hp_wavelet_levels);
        mra_hp = modwtmra_custom(wpt_hp, wavelet_type);
        clear wpt_hp;
    end
end

% Identify wavelet bands to remove based on lowcut_frequency (VECTORIZED)
srate = EEGavRef.srate;
num_bands_hp = size(mra_hp, 1);
% Vectorize: compute all upper bounds at once
upper_bounds = srate ./ (2.^(1:num_bands_hp));
bands_to_zero = find(upper_bounds <= lowcut_frequency);
if ~isempty(bands_to_zero)
    mra_hp(bands_to_zero, :, :) = 0; % Remove bands below the cutoff
end

% Reconstruct the high-passed signal
EEGavRef.data = squeeze(sum(mra_hp, 1))';
clear mra_hp

    % ------------------ GEDAI ------------------------------

    disp([newline 'SENSAI threshold detection...please wait']);
    broadband_optimization_type = 'parabolic';
    broadband_artifact_threshold_type = 'auto-';
    broadband_minThreshold = 0;
    broadband_maxThreshold = 12;
    [cleaned_broadband_data, ~, broadband_sensai, broadband_thresh, broadband_ENOVA, broadband_SSI_angles] = GEDAI_per_band(double(EEGavRef.data), EEGavRef.srate, EEGavRef.chanlocs, broadband_artifact_threshold_type, broadband_epoch_size, refCOV, broadband_optimization_type, parallel, signal_type, broadband_minThreshold, broadband_maxThreshold);
    SENSAI_score_per_band = broadband_sensai;
    artifact_threshold_per_band = broadband_thresh;
    ENOVA_per_band = broadband_ENOVA;

    % Initialize storage for per-band visualization
    SSI_angles_all_bands = cell(1, 1);
    SSI_angles_all_bands{1} = broadband_SSI_angles;
    SSI_labels_all_bands = {'Broadband'};




%% Second pass: Wavelet decomposition and per-band denoising
% MEMORY OPTIMIZED: Use incremental band processing instead of full decomposition
unfiltered_data = cleaned_broadband_data';
wavelet_type = 'haar';
number_of_wavelet_bands = 9; % Default number of wavelet bands = 9

% OPTIMIZATION: Eliminated full wpt_EEG storage - bands will be extracted incrementally
number_of_discrete_wavelet_bands = number_of_wavelet_bands;
% Actual decomposition level needed to create number_of_discrete_wavelet_bands
actual_decomposition_level = number_of_discrete_wavelet_bands - 1;  % MODWT creates level+1 bands

% MEMORY OPTIMIZED: Clear source data immediately (no longer needed for full decomposition)
clear cleaned_broadband_data;

% Pre-calculate center frequencies for each MRA wavelet band
srate = EEGavRef.srate;
center_frequencies = zeros(1, number_of_discrete_wavelet_bands);
lower_frequencies = zeros(1, number_of_discrete_wavelet_bands);
upper_frequencies = zeros(1, number_of_discrete_wavelet_bands);
for f = 1:number_of_discrete_wavelet_bands
    % The passband for MRA band 'f' is approx. [Fs/(2^(f+1)), Fs/(2^f)]
    lower_bound = srate / (2^(f + 1));
    upper_bound = srate / (2^f);
    center_frequencies(f) = (lower_bound + upper_bound) / 2;
    lower_frequencies(f) = lower_bound; 
    upper_frequencies(f) = upper_bound;
end


lowest_wavelet_bands_to_exclude = sum(upper_frequencies <= lowcut_frequency); 
num_bands_to_process = number_of_discrete_wavelet_bands - lowest_wavelet_bands_to_exclude;

% --- Check if data is long enough for the lowest frequency epoch size---
if num_bands_to_process > 0
    lowest_band_to_process_idx = num_bands_to_process;
    epoch_size_lowest_band = epoch_size_in_cycles / lower_frequencies(lowest_band_to_process_idx);
    required_samples = epoch_size_lowest_band * srate;

    while required_samples > size(EEGavRef.data, 2) && num_bands_to_process > 0
        warning('GEDAI:InsufficientData', 'EEG data length is too short for the epoch size required by the lowest frequency band (%g Hz). Increasing lowcut_frequency.', center_frequencies(lowest_band_to_process_idx));
        lowcut_frequency = upper_frequencies(lowest_band_to_process_idx);
        lowest_wavelet_bands_to_exclude = sum(upper_frequencies <= lowcut_frequency);
        num_bands_to_process = number_of_discrete_wavelet_bands - lowest_wavelet_bands_to_exclude;
        
        lowest_band_to_process_idx = num_bands_to_process;
        epoch_size_lowest_band = epoch_size_in_cycles / lower_frequencies(lowest_band_to_process_idx);
        required_samples = epoch_size_lowest_band * srate;
    end
end

%%  Define Frequency-Dependent Epoch Sizes ---

% Calculate the ideal epoch size for each band based on the rule
epoch_sizes_per_wavelet_band = epoch_size_in_cycles ./ lower_frequencies;

% --- Display wavelet band-widths and epoch sizes ---
% disp(' ');  
left_margin = '  '; 
header1 = 'Wavelet Center Freq (Hz)';
header2 = 'Epoch Size (s)';
str_freqs = num2str(center_frequencies(1:num_bands_to_process)', '%.2g');
str_epochs = num2str(epoch_sizes_per_wavelet_band(1:num_bands_to_process)', '%.2g');
col1_width = max(length(header1), size(str_freqs, 2));
col2_width = max(length(header2), size(str_epochs, 2));

disp([newline 'Excluding ', num2str(lowest_wavelet_bands_to_exclude), ' wavelet bands with upper frequency < ' num2str(lowcut_frequency) ' Hz.']);


% Correct each epoch size to ensure it corresponds to an even number of samples
for f = 1:num_bands_to_process
    ideal_samples = epoch_sizes_per_wavelet_band(f) * srate;
    rounded_samples = round(ideal_samples);
    if rem(rounded_samples, 2) ~= 0
        % If odd, choose the nearest even number
        if abs(ideal_samples - (rounded_samples - 1)) < abs(ideal_samples - (rounded_samples + 1))
            final_samples = rounded_samples - 1;
        else
            final_samples = rounded_samples + 1;
        end
    else
        final_samples = rounded_samples;
    end
    epoch_sizes_per_wavelet_band(f) = final_samples / srate;
end

%% Denoise each wavelet band
% MEMORY OPTIMIZED: Get dimensions from unfiltered data
[num_samples, num_channels] = size(unfiltered_data);

% MEMORY OPTIMIZED: Use 2D accumulator with correct type
% Pre-allocate with same precision as input data
wavelet_band_filtered_data = zeros(num_channels, num_samples, 'like', unfiltered_data);
success_parallel = false;

if parallel
    try
        temp_sensai_scores = zeros(1, num_bands_to_process);
        temp_thresholds = zeros(1, num_bands_to_process);
        temp_enova_scores = zeros(1, num_bands_to_process);
        temp_SSI_angles = cell(1, num_bands_to_process);
        temp_SSI_labels = cell(1, num_bands_to_process);
        
        % MEMORY OPTIMIZED: Incremental band extraction in parallel
        parfor f = 1:num_bands_to_process
            % Extract single band on-the-fly (no full wpt_EEG storage)
            wavelet_data_band = modwt_single_band(unfiltered_data, wavelet_type, actual_decomposition_level, f)';
            
            current_epoch_size = epoch_sizes_per_wavelet_band(f);
            
            % Determine minThreshold based on signal type and frequency
            current_center_freq = center_frequencies(f);
            current_minThreshold = 0;
            if (current_center_freq >= 7 && current_center_freq <= 13)
                current_minThreshold = -6;
            end

            try
                 [cleaned_band_data, ~, temp_sensai, temp_thresh, temp_enova_val, SSI_angles_val] = GEDAI_per_band(wavelet_data_band, srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold);
            catch ME
                 % If OOM or other memory error, try single precision
                 warning('GEDAI_per_band failed for band %d: %s. Retrying with single precision...', f, ME.message);
                 [cleaned_band_data, ~, temp_sensai, temp_thresh, temp_enova_val, SSI_angles_val] = GEDAI_per_band(single(wavelet_data_band), srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold);
            end
            
            % RAM OPTIMIZATION: Accumulate directly using a reduction variable (avoids massive cell array copies)
            wavelet_band_filtered_data = wavelet_band_filtered_data + cleaned_band_data;
            temp_sensai_scores(f) = temp_sensai;
            temp_thresholds(f) = temp_thresh;
            temp_enova_scores(f) = temp_enova_val;
            temp_SSI_angles{f} = SSI_angles_val;
            temp_SSI_labels{f} = [num2str(current_center_freq, '%.2g') ' Hz'];
        end
        
        SENSAI_score_per_band = [SENSAI_score_per_band, temp_sensai_scores];
        artifact_threshold_per_band = [artifact_threshold_per_band, temp_thresholds];
        ENOVA_per_band = [ENOVA_per_band, temp_enova_scores];
        SSI_angles_all_bands = [SSI_angles_all_bands, temp_SSI_angles];
        SSI_labels_all_bands = [SSI_labels_all_bands, temp_SSI_labels];
        success_parallel = true;
    catch 
        warning('Parallel processing failed: %s. Switching to double precision non-parallel processing.');
    end
end

if ~parallel || ~success_parallel
    success_serial = false;
    if parallel && ~success_parallel
         disp('Executing fallback: Double Precision Non-Parallel Processing...');
    end
    
    try
        % MEMORY OPTIMIZED: Sequential processing with incremental band extraction
        for f = 1:num_bands_to_process
            % Extract single band on-the-fly (no full wpt_EEG storage)
            wavelet_data_band = modwt_single_band(unfiltered_data, wavelet_type, actual_decomposition_level, f)';
            
            current_epoch_size = epoch_sizes_per_wavelet_band(f);
            
            % Determine minThreshold based on signal type and frequency
            current_center_freq = center_frequencies(f);
            current_minThreshold = 0;
            if (current_center_freq >= 7 && current_center_freq <= 13)
                current_minThreshold = -6;
            end
            
            try
              disp(['processing wavelet band = ' num2str(f)])   
              [cleaned_band_data, ~, sensai_val, thresh_val, enova_val, SSI_angles_val] = GEDAI_per_band(double(wavelet_data_band), srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold);
            
            catch ME
                warning('GEDAI_per_band failed for band %d: %s. Retrying with single precision...', f, ME.message);
                [cleaned_band_data, ~, sensai_val, thresh_val, enova_val, SSI_angles_val] = GEDAI_per_band(single(wavelet_data_band), srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold);
            end
            
            % MEMORY OPTIMIZED: Accumulate directly into 2D array
            wavelet_band_filtered_data = wavelet_band_filtered_data + cleaned_band_data;
            SENSAI_score_per_band(f+1) = sensai_val;
            artifact_threshold_per_band(f+1) = thresh_val;
            ENOVA_per_band(f+1) = enova_val;
            SSI_angles_all_bands{f+1} = SSI_angles_val;
            SSI_labels_all_bands{f+1} = [num2str(current_center_freq, '%.2g') ' Hz'];
            
            % MEMORY OPTIMIZED: Clear band data immediately
            clear wavelet_data_band cleaned_band_data;
        end
        success_serial = true;
    catch
        warning('Double Precision Non-Parallel processing failed: %s. Switching to LAST RESORT: Single Precision Non-Parallel Processing.');
    end
    
    if ~success_serial
         disp('Executing Last Resort: Single Precision Non-Parallel Processing...');
         for f = 1:num_bands_to_process
            % Extract single band on-the-fly (no full wpt_EEG storage)
            wavelet_data_band = modwt_single_band(single(unfiltered_data), wavelet_type, actual_decomposition_level, f)';
            current_epoch_size = epoch_sizes_per_wavelet_band(f);
            
            % Determine minThreshold based on signal type and frequency
            current_center_freq = center_frequencies(f);
            current_minThreshold = 0;
            if strcmpi(signal_type, 'meg') && (current_center_freq >= 7 && current_center_freq <= 13)
                current_minThreshold = -6;
            end
            
            [cleaned_band_data, ~, sensai_val, thresh_val, enova_val, SSI_angles_val] = GEDAI_per_band(single(wavelet_data_band), srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold);
            disp(['processing wavelet band (single) = ' num2str(f)])
            
            % MEMORY OPTIMIZED: Accumulate directly into 2D array
            wavelet_band_filtered_data = wavelet_band_filtered_data + cleaned_band_data;
            SENSAI_score_per_band(f+1) = sensai_val;
            artifact_threshold_per_band(f+1) = thresh_val;
            ENOVA_per_band(f+1) = enova_val;
            SSI_angles_all_bands{f+1} = SSI_angles_val;
            SSI_labels_all_bands{f+1} = [num2str(current_center_freq, '%.2g') ' Hz'];
            
            % MEMORY OPTIMIZED: Clear band data immediately
            clear wavelet_data_band cleaned_band_data;
         end
    end
end

% MEMORY OPTIMIZED: Clear unfiltered data after all wavelet processing
clear unfiltered_data;

%% Finalization: Reconstruct EEG and calculate final scores
% MEMORY OPTIMIZED: Data already accumulated in 2D array, no summation needed
EEGclean = EEGavRef;
EEGclean.data = wavelet_band_filtered_data;  % Already accumulated
% Create artifact structure
EEGartifacts = EEGclean;
EEGartifacts.data = EEGavRef.data(:, 1:size(EEGclean.data, 2)) - EEGclean.data;

% Calculate composite SENSAI score for epoch rejection
noise_multiplier = 1;
[SENSAI_score, ~, ~, mean_ENOVA, ENOVA_per_epoch] = SENSAI_basic(double(EEGclean.data), double(EEGartifacts.data), EEGavRef.srate, broadband_epoch_size, refCOV, noise_multiplier, signal_type);

% Store original epoch count for rejection statistics
original_total_epochs = length(ENOVA_per_epoch);
epochs_to_remove = find(ENOVA_per_epoch > ENOVA_threshold);
regions = [];
if ~isempty(epochs_to_remove)
    epoch_samples = round(broadband_epoch_size * EEGavRef.srate);
    regions = zeros(length(epochs_to_remove), 2);
    for i = 1:length(epochs_to_remove)
        epoch = epochs_to_remove(i);
        start_sample = (epoch - 1) * epoch_samples + 1;
        end_sample = epoch * epoch_samples;
        if end_sample > size(EEGclean.data, 2)
            end_sample = size(EEGclean.data, 2);
        end
        regions(i,:) = [start_sample end_sample];
    end
end

tEnd = toc(tStart);


% Generate command history
if ~ischar(ref_matrix_type)
    ref_matrix_type = 'custom';
end
com = sprintf('EEG = GEDAI(EEG, ''%s'', %s,  %s, ''%s'', %d,  %d, %s, ''%s'');', ...
    artifact_threshold_type, num2str(epoch_size_in_cycles), num2str(lowcut_frequency), ref_matrix_type, parallel, visualize_artifacts, num2str(ENOVA_threshold), signal_type);

if visualize_artifacts
    EEGclean_for_vis = EEGclean;
    if ~isempty(regions)
        clean_sample_mask = true(1, EEGclean_for_vis.pnts);
        for i = 1:size(regions, 1)
            clean_sample_mask(regions(i,1):regions(i,2)) = false;
        end
        EEGclean_for_vis.etc.clean_sample_mask = clean_sample_mask;
        EEGclean_for_vis.data = EEGclean_for_vis.data(:, clean_sample_mask);
        EEGclean_for_vis.pnts = size(EEGclean_for_vis.data, 2);
    end
    vis_artifacts(EEGclean_for_vis, EEGavRef, 'ScaleBy', 'noscale', 'YScaling', 3*mad(EEGavRef.data(:)));
end

if ~isempty(regions)
    disp([newline 'Removing bad epochs...']);
    
    % Manual implementation of eeg_eegrej to avoid eeg_checkset issues
    samples_to_keep = true(1, EEGclean.pnts);
    for i = 1:size(regions, 1)
        start_idx = round(regions(i,1));
        end_idx = round(regions(i,2));
        if start_idx > 0 && end_idx <= EEGclean.pnts
            samples_to_keep(start_idx:end_idx) = false;
        end
    end

    % --- Apply windowing/tapering to smooth discontinuities at the seams after epoch rejection ---
    taper_duration = 0.05; % 50 ms
    taper_points = round(taper_duration * EEGclean.srate);
    % Create a cosine taper (half-Hanning)
    % We want 0 to 1 over 'taper_points'. 
    % Hanning is (1 - cos(phi))/2. 
    % phi=0 -> 0. phi=pi -> 1.
    taper_phase = linspace(0, pi, taper_points);
    taper_attack = (1 - cos(taper_phase)) / 2; % Rise 0 to 1
    taper_decay = fliplr(taper_attack);        % Fall 1 to 0

    % Find transitions
    % diff = -1: Keep -> Reject (End of valid segment) -> Apply Decay
    decay_indices = find(diff(samples_to_keep) == -1);
    
    % diff = 1: Reject -> Keep (Start of valid segment) -> Apply Attack
    attack_indices = find(diff(samples_to_keep) == 1) + 1;
    
    % Apply Decay (Fade Out)
    for idx = decay_indices
        s_start = max(1, idx - taper_points + 1);
        s_end = idx;
        len = s_end - s_start + 1;
        
        current_taper = taper_decay(end-len+1:end); % Match length
        EEGclean.data(:, s_start:s_end) = EEGclean.data(:, s_start:s_end) .* current_taper; 
        EEGartifacts.data(:, s_start:s_end) = EEGartifacts.data(:, s_start:s_end) .* current_taper;
    end
    
    % Apply Attack (Fade In)
    for idx = attack_indices
        s_start = idx;
        s_end = min(EEGclean.pnts, idx + taper_points - 1);
        len = s_end - s_start + 1;
        
        current_taper = taper_attack(1:len);
        EEGclean.data(:, s_start:s_end) = EEGclean.data(:, s_start:s_end) .* current_taper;
        EEGartifacts.data(:, s_start:s_end) = EEGartifacts.data(:, s_start:s_end) .* current_taper;
    end
    % -----------------------------------------------------------------------

    % Apply mask to EEGclean
    EEGclean.data = EEGclean.data(:, samples_to_keep);
    EEGclean.pnts = size(EEGclean.data, 2);
    EEGclean.xmax = EEGclean.xmin + (EEGclean.pnts-1)/EEGclean.srate;
    if EEGclean.pnts > 1
        EEGclean.times = linspace(EEGclean.xmin*1000, EEGclean.xmax*1000, EEGclean.pnts);
    else
        EEGclean.times = EEGclean.xmin*1000;
    end
    
    % Apply mask to EEGartifacts
    EEGartifacts.data = EEGartifacts.data(:, samples_to_keep);
    EEGartifacts.pnts = size(EEGartifacts.data, 2);
    EEGartifacts.xmax = EEGartifacts.xmin + (EEGartifacts.pnts-1)/EEGartifacts.srate;
    if EEGartifacts.pnts > 1
        EEGartifacts.times = linspace(EEGartifacts.xmin*1000, EEGartifacts.xmax*1000, EEGartifacts.pnts);
    else
        EEGartifacts.times = EEGartifacts.xmin*1000;
    end
end

% Calculate final SENSAI score (after potential epoch rejection)

[SENSAI_score, ~, ~, mean_ENOVA, ENOVA_per_epoch] = SENSAI_basic(double(EEGclean.data), double(EEGartifacts.data), EEGavRef.srate, broadband_epoch_size, refCOV, noise_multiplier, signal_type);

% disp([newline 'SENSAI score: ' num2str(round(SENSAI_score, 2, 'significant'))]);
% disp(['Mean ENOVA: ' num2str(round(mean_ENOVA, 2, 'significant'))]);

% Use original epoch count for rejection statistics (before rejection)
num_rejected = length(epochs_to_remove);
if original_total_epochs > 0
    percentage_rejected = (num_rejected / original_total_epochs) * 100;
else
    percentage_rejected = 0;
end
% disp(['Bad epochs rejected: ' num2str(round(percentage_rejected,1)) ' % (' num2str(num_rejected) ' out of ' num2str(original_total_epochs) ' epochs)']);

% --- Summarized Output Table (including ENOVA) ---
disp(' '); 
left_margin = '  '; 
header1 = 'Wavelet Center Freq (Hz)';
header2 = 'Epoch Size (s)';
header3 = 'ENOVA (%)';

% Combine frequencies and epochs for display, including Broadband
% Broadband is index 1 in the arrays, usually displayed first
% We can label frequency as "Broadband" or Inf/NaN
freq_str_cell = cell(1, num_bands_to_process + 1);
freq_str_cell{1} = 'Broadband';
for i = 1:num_bands_to_process
    freq_str_cell{i+1} = [num2str(center_frequencies(i), '%.2g') ' Hz'];
end

epoch_str_cell = cell(1, num_bands_to_process + 1);
epoch_str_cell{1} = [num2str(broadband_epoch_size, '%.2g') ' s'];
for i = 1:num_bands_to_process
    epoch_str_cell{i+1} = [num2str(epoch_sizes_per_wavelet_band(i), '%.2g') ' s'];
end

enova_str_cell = cell(1, num_bands_to_process + 1);
% ENOVA_per_band contains [Broadband, Band1, Band2, ...] if processed sequentially
% Ensure correct indexing
for i = 1:length(ENOVA_per_band)
    enova_str_cell{i} = [num2str(round(ENOVA_per_band(i) * 100), '%.0f') ' %'];
end

% Determine column widths
max_freq_width = max(cellfun(@length, freq_str_cell));
col1_width = max(length(header1), max_freq_width) + 2; % Add padding
max_epoch_width = max(cellfun(@length, epoch_str_cell));
col2_width = max(length(header2), max_epoch_width) + 2; % Add padding
max_enova_width = max(cellfun(@length, enova_str_cell));
col3_width = max(length(header3), max_enova_width) + 2; % Add padding

% Centering helper function
center_text = @(str, width) [repmat(' ', 1, floor((width-length(str))/2)), str, repmat(' ', 1, ceil((width-length(str))/2))];

fprintf('%s%s | %s | %s\n', left_margin, center_text(header1, col1_width), center_text(header2, col2_width), center_text(header3, col3_width));
fprintf('%s%s-|-%s-|-%s\n', left_margin, repmat('-', 1, col1_width), repmat('-', 1, col2_width), repmat('-', 1, col3_width));

for i = 1:length(freq_str_cell)
    fprintf('%s%s | %s | %s\n', left_margin, center_text(freq_str_cell{i}, col1_width), center_text(epoch_str_cell{i}, col2_width), center_text(enova_str_cell{i}, col3_width));
end
disp(' ');

disp([newline 'SENSAI score: ' num2str(round(SENSAI_score, 2, 'significant'))]);
disp(['Mean ENOVA: ' num2str(round(mean_ENOVA*100, 2, 'significant')) ' %']);
disp(['Bad epochs rejected: ' num2str(round(percentage_rejected,1)) ' % (' num2str(num_rejected) ' out of ' num2str(original_total_epochs) ' epochs)']);
disp(['Elapsed time: ' num2str(round(tEnd, 2, 'significant')) ' seconds' newline]);

% Store GEDAI variables in EEG.etc.GEDAI
EEGclean.etc.GEDAI.SENSAI_score = SENSAI_score;
EEGclean.etc.GEDAI.SENSAI_score_per_band = SENSAI_score_per_band;
EEGclean.etc.GEDAI.artifact_threshold_per_band = artifact_threshold_per_band;
EEGclean.etc.GEDAI.mean_ENOVA = mean_ENOVA;
EEGclean.etc.GEDAI.ENOVA_per_band = ENOVA_per_band;
EEGclean.etc.GEDAI.ENOVA_per_epoch = ENOVA_per_epoch;
EEGclean.etc.GEDAI.epochs_rejected = num_rejected;
EEGclean.etc.GEDAI.total_epochs = original_total_epochs;
EEGclean.etc.GEDAI.percentage_rejected = percentage_rejected;
if exist('samples_to_keep', 'var')
    EEGclean.etc.GEDAI.samples_to_keep = samples_to_keep;
else
    EEGclean.etc.GEDAI.samples_to_keep = true(1, original_total_epochs * round(broadband_epoch_size * EEGavRef.srate)); 
    % Note: The above calculation might be slightly off if rounding happened differently for 'pnts'.
    % Safer to use current pnts if no rejection happened:
    EEGclean.etc.GEDAI.samples_to_keep = true(1, size(EEGclean.data, 2));
end


    % --- Manifold Classification (Broadband) BEFORE Cleaning ---
    if ~isempty(refCOV)
        %fprintf('\nGenerating SENSAI Plot for Broadband data...\n');
        
        % Calculate Covariance of every epoch for broadband data BEFORE cleaning
        % Using the same padding logic as GEDAI_per_band for consistency
        pnts_original = size(EEGavRef.data, 2);
        epoch_samples = round(EEGavRef.srate * broadband_epoch_size);
        eeg_data_temp = EEGavRef.data;
        
        remainder = rem(pnts_original, epoch_samples);
        if remainder ~= 0
            samples_to_pad = epoch_samples - remainder;
            reflection_segment = eeg_data_temp(:, end-samples_to_pad+1:end);
            eeg_data_temp = [eeg_data_temp, fliplr(reflection_segment)];
        end
        
        EEGdata_epoched = reshape(eeg_data_temp, size(eeg_data_temp, 1), epoch_samples, []);
        num_epochs = size(EEGdata_epoched, 3);
        COV_emp_array_before = cell(num_epochs, 1);
        for epo = 1:num_epochs
            COV_emp_array_before{epo} = cov(EEGdata_epoched(:,:,epo)');
        end
    end


    % --- Manifold Classification (Broadband) AFTER Cleaning ---
    if visualize_manifold && ~isempty(refCOV)
        % fprintf('\nGenerating SENSAI Plot for Final Reconstructed Data (Before/After)...\n');
        
        % Calculate Covariance of every epoch for final data AFTER full cleaning pipeline
        eeg_data_temp = EEGclean.data;
        artifact_data_temp = EEGartifacts.data;
        
        if exist('remainder', 'var') && remainder ~= 0
            reflection_segment = eeg_data_temp(:, end-samples_to_pad+1:end);
            eeg_data_temp = [eeg_data_temp, fliplr(reflection_segment)];

            reflection_segment_artifact = artifact_data_temp(:, end-samples_to_pad+1:end);
            artifact_data_temp = [artifact_data_temp, fliplr(reflection_segment_artifact)];
        end
        
        EEGdata_epoched = reshape(eeg_data_temp, size(eeg_data_temp, 1), epoch_samples, []);
        EEGartifacts_epoched = reshape(artifact_data_temp, size(artifact_data_temp, 1), epoch_samples, []);
        
        COV_emp_array_after = cell(num_epochs, 1);
        COV_emp_array_artifacts = cell(num_epochs, 1);
        for epo = 1:num_epochs
            COV_emp_array_after{epo} = cov(EEGdata_epoched(:,:,epo)');
            COV_emp_array_artifacts{epo} = cov(EEGartifacts_epoched(:,:,epo)');
        end
        
        SENSAI_visualization(refCOV, COV_emp_array_before, COV_emp_array_after, COV_emp_array_artifacts);
        
        % Generate tiled visualization for all bands
        SENSAI_visualization_tiled(refCOV, SSI_angles_all_bands, SSI_labels_all_bands);
    end

% Add command history to EEGLAB structure
if exist('eegh', 'file')
    EEGclean = eegh(com, EEGclean);
end
end