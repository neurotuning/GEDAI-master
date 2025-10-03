%   GEDAI() - This is the main function used to denoise EEG data using
%   generalized eigenvalue decomposition coupled with an EEG leadfield matrix
%
%   GEDAI (Generalized Eigenvalue Deartifacting Instrument)
%
% Usage:
% Example 1: Using all default values
%    >>  [EEG] = GEDAI(EEG);
%
% Example 2: Defining all parameters (default settings)
%    >>  [EEG] = GEDAI(EEG, 'auto', 1.0, 'precomputed', true, false);
%
% Example 3: Using a "custom" [channel x channel] reference matrix 
%    >>  [EEG] = GEDAI(EEG, auto', 1.0, your_refCOV);
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
%                                 Default is "auto" which strikes a balance.
%                             
%   epoch_size                  - Epoch size in seconds. Default value is 1 second
%                                 but any value is possible as long as the number
%                                 of samples in the epoch is higher than the sampling rate. 
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
% 
%   com                     - output logging to EEG.history

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

function [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, com]=GEDAI(EEGin,artifact_threshold_type, epoch_size,ref_matrix_type,parallel,visualize_artifacts)

arguments
    EEGin struct;
    artifact_threshold_type = 'auto' ;
    epoch_size = 1;
    ref_matrix_type = 'precomputed' ;
    parallel = true ;
    visualize_artifacts = false ;
end

p = fileparts(which('GEDAI'));
addpath(fullfile(p, 'auxiliaries'));

tStart = tic;  

% -- Ensure epoch size results in an even number of samples
if rem(epoch_size*EEGin.srate, 2) ~= 0
    ideal_total_samples_double = epoch_size * EEGin.srate;
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
    epoch_size = target_total_samples_int / EEGin.srate;
end

%% Create Reference Covariance Matrix (refCOV)
if ~ischar(ref_matrix_type)
    refCOV = ref_matrix_type; % Use custom covariance matrix
    disp([newline 'Using custom covariance matrix']);
else
    switch ref_matrix_type
        case 'precomputed'
            disp([newline 'GEDAI Leadfield model: BEM precomputed'])
            L=load('fsavLEADFIELD_4_GEDAI.mat');
            electrodes_labels = {EEGin.chanlocs.labels};
            template_electrode_labels = {L.leadfield4GEDAI.electrodes.Name};
            [~, chanidx] = ismember(lower(electrodes_labels), lower(template_electrode_labels));
            if any(chanidx == 0)
                error('Electrode labels not found. Select "interpolated" leadfield matrix for non-standard locations.');
            end
            refCOV = L.leadfield4GEDAI.gram_matrix_avref(chanidx,chanidx);
        case 'interpolated'
            disp([newline 'GEDAI Leadfield model: BEM interpolated'])
            L=load('fsavLEADFIELD_4_GEDAI.mat');
            leadfield_EEG = L.leadfield4GEDAI.EEG;
            leadfield_EEG.data = L.leadfield4GEDAI.Gain - mean(L.leadfield4GEDAI.Gain, 1); % Average reference
            interpolated_EEG = interp_mont_GEDAI(leadfield_EEG, EEGin.chanlocs);
            refCOV = interpolated_EEG.data*interpolated_EEG.data';
    end
end

%% Pre-processing
EEGin = GEDAI_nonRankDeficientAveRef(EEGin);
EEG_original_data = EEGin.data; % Keep a copy of original data for visualization

%% First pass: Broadband denoising
disp([newline 'Artifact threshold detection...please wait']);
broadband_optimization_type = 'parabolic';
broadband_artifact_threshold_type = 'auto-';

[cleaned_broadband_data, ~, broadband_sensai, broadband_thresh] = GEDAI_per_band(double(EEGin.data), EEGin.srate, EEGin.chanlocs, broadband_artifact_threshold_type, epoch_size, refCOV, broadband_optimization_type, parallel);

% Initialize the output arrays with the broadband results
SENSAI_score_per_band = broadband_sensai;
artifact_threshold_per_band = broadband_thresh;


%% Second pass: Wavelet decomposition and per-band denoising
unfiltered_data = cleaned_broadband_data';
number_of_wavelet_levels = 3;
number_of_wavelet_bands = 2^number_of_wavelet_levels + 1;
wavelet_type = 'haar';

wpt_EEG = modwt(unfiltered_data, wavelet_type, number_of_wavelet_bands);
wpt_EEG = modwtmra(wpt_EEG, wavelet_type); % wavelet bands x samples x channels

number_of_discrete_wavelet_bands = size(wpt_EEG, 1);
lowest_wavelet_bands_to_exclude = ceil(600/EEGin.srate);
num_bands_to_process = number_of_discrete_wavelet_bands - lowest_wavelet_bands_to_exclude;

num_channels = size(cleaned_broadband_data, 1);
num_samples = size(cleaned_broadband_data, 2);
wavelet_band_filtered_data = zeros(num_bands_to_process, num_channels, num_samples);


%% Denoise each wavelet band
if parallel
    % Create temporary storage for parfor results
    temp_sensai_scores = zeros(1, num_bands_to_process);
    temp_thresholds = zeros(1, num_bands_to_process);
    
    parfor f = 1:num_bands_to_process
        % Get the data for the current band
        wavelet_data_band = transpose(squeeze(wpt_EEG(f,:,:)));

        % Call the processing function directly inside the loop
        [cleaned_band_data, ~, temp_sensai, temp_thresh] = GEDAI_per_band(double(wavelet_data_band), EEGin.srate, EEGin.chanlocs, artifact_threshold_type, epoch_size, refCOV, 'parabolic', false);
        
        % Assign results to sliced output variables
        wavelet_band_filtered_data(f, :,:) = cleaned_band_data;
        temp_sensai_scores(f) = temp_sensai;
        temp_thresholds(f) = temp_thresh;
    end
    
    % After the loop, append the results to the main arrays
    SENSAI_score_per_band = [SENSAI_score_per_band, temp_sensai_scores];
    artifact_threshold_per_band = [artifact_threshold_per_band, temp_thresholds];
    
else % Non-parallel version
    for f = 1:num_bands_to_process
        disp([newline 'wavelet band = ' num2str(f)])
        wavelet_data_band = transpose(squeeze(wpt_EEG(f,:,:)));
        
        % Call the processing function directly
        [cleaned_band_data, ~, sensai_val, thresh_val] = GEDAI_per_band(wavelet_data_band, EEGin.srate, EEGin.chanlocs, artifact_threshold_type, epoch_size, refCOV, 'parabolic', false);
        
        wavelet_band_filtered_data(f, :,:) = cleaned_band_data;
        SENSAI_score_per_band(f+1) = sensai_val;
        artifact_threshold_per_band(f+1) = thresh_val;
    end
end


%% Finalization: Reconstruct EEG and calculate final scores
% Reconstruct EEG from cleaned wavelet bands
EEGclean = EEGin;
EEGclean.data = squeeze(sum(wavelet_band_filtered_data, 1));

% Ensure original data has same length as cleaned data for artifact calculation
EEGin.data = EEGin.data(:, 1:size(EEGclean.data, 2));

% Create artifact structure
EEGartifacts = EEGclean;
EEGartifacts.data = EEGin.data - EEGclean.data;

% Calculate composite SENSAI score
noise_multiplier = 1;
[SENSAI_score] = SENSAI_basic(double(EEGclean.data), double(EEGartifacts.data), EEGin.srate, epoch_size, refCOV, noise_multiplier);
tEnd = toc(tStart);

disp([newline 'SENSAI score: ' num2str(round(SENSAI_score, 2, "significant"))]);
disp(['Elapsed time: ' num2str(round(tEnd, 2, "significant")) ' seconds']);

% Generate command history
com = sprintf('EEG = GEDAI(EEG, ''%s'', %s, ''%s'', %d, %d);', ...
    artifact_threshold_type, num2str(epoch_size), ref_matrix_type, parallel, visualize_artifacts);
EEGclean = eegh(com, EEGclean);

if visualize_artifacts
    vis_artifacts(EEGclean, EEGin, 'ScaleBy', 'noscale', 'YScaling', 3*mad(EEG_original_data(:)), 'show_removed_portions', false);
end
end
