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

function [cleaned_data, artifacts_data, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold_in, refCOV, Eval, Evec)
%   This GEDAI function reconstructs the signal after removing artifactual components

% --- PRE-ALLOCATION ---
num_chans = size(Eval, 1);
num_epochs = size(Eval, 3);
% Pre-allocate the array to its full size with the correct complex type.
all_diagonals = complex(zeros(num_chans * num_epochs, 1));
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
epoch_samples = srate * epoch_size;
artifacts = complex(zeros(size(EEGdata_epoched)));
cleaned_epoched_data = complex(zeros(size(EEGdata_epoched)));
cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
half_epoch = epoch_samples/2;

for i = 1:num_epochs

    component_spatial_filter = Evec(:,:,i);
    
    for j = 1:num_chans
        if abs(Eval(j,j,i)) < exp(Treshold1 - 100)
            component_spatial_filter(:,j) = 0;
        end
    end
    
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
cleaned_data = real(reshape(cleaned_epoched_data, num_chans, []));
artifacts_data = real(reshape(artifacts, num_chans, []));
artifact_threshold_out = artifact_threshold_in;

end
