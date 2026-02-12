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

function [cleaned_data, artifacts_data, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold_in, refCOV, Eval, Evec, cosine_weights)
%   This GEDAI function reconstructs the signal after removing artifactual components

% --- PRE-ALLOCATION ---
num_chans = size(Eval, 1);
num_epochs = size(Eval, 3);
% Pre-allocate the array 
all_diagonals = zeros(num_chans * num_epochs, 1, 'like', Eval);
for i = 1:num_epochs
    start_idx = (i-1) * num_chans + 1;
    end_idx = i * num_chans;
    all_diagonals(start_idx:end_idx) = diag(Eval(:,:,i));
end
% Use the magnitude (a real value) for all subsequent calculations.
magnitudes = abs(all_diagonals);
log_Eig_val_all = log(magnitudes(magnitudes > 0)) + 100;


%% Artifacting multiplication factor T1
correction_factor = 1.00;
T1 = correction_factor * (105 - artifact_threshold_in) / 100;

%% Defining artifact threshold with Probability Integral Transform (PIT) fitting
original_data = unique(log_Eig_val_all);
[f,x] = ecdf(original_data);
[unique_x, ia, ~] = unique(original_data);
unique_f = f(ia);
transformed_data = interp1(unique_x, unique_f, original_data, 'linear', 'extrap');
upper_PIT_threshold = 0.95;
outliers = original_data(transformed_data > upper_PIT_threshold);
Treshold1 = T1 * min(outliers);

%% Cleaning EEG by removing outlying GEVD components
epoch_samples = round(srate * epoch_size);
artifacts = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
cleaned_epoched_data = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
if nargin < 8 || isempty(cosine_weights)
    cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
end
half_epoch = floor(epoch_samples/2);

for i = 1:num_epochs
    component_spatial_filter = Evec(:,:,i);
    
    % --- OPTIMIZATION START ---
    % 1. Create a logical mask of indices to zero out
    % This replaces the entire 'for j' loop with one line
    bad_indices = abs(diag(Eval(:,:,i))) < exp(Treshold1 - 100);
    
    % 2. Apply the mask (Vectorized)
    component_spatial_filter(:, bad_indices) = 0;
    % --- OPTIMIZATION END ---

    artifacts_timecourses = component_spatial_filter' * EEGdata_epoched(:,:,i);    
    Signal_to_remove = Evec(:,:,i)' \ artifacts_timecourses;
    
    artifacts(:, :, i) = Signal_to_remove;
    cleaned_epoch = EEGdata_epoched(:,:,i) - Signal_to_remove;
    
    % Apply cosine windowing to mitigate edge effects from epoching
    if i == 1
        cleaned_epoch(:, half_epoch+1:end) = cleaned_epoch(:, half_epoch+1:end) .* cosine_weights(:, half_epoch+1:end);
    elseif i == num_epochs
        cleaned_epoch(:, 1:half_epoch) = cleaned_epoch(:, 1:half_epoch) .* cosine_weights(:, 1:half_epoch);
    else
        cleaned_epoch = cleaned_epoch .* cosine_weights;
    end
    
    cleaned_epoched_data(:,:,i) = cleaned_epoch;
end

% Reshape data back to continuous form and return outputs
cleaned_data = reshape(cleaned_epoched_data, num_chans, []);
artifacts_data = reshape(artifacts, num_chans, []);
artifact_threshold_out = artifact_threshold_in;

end