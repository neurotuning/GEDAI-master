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

<<<<<<< Updated upstream
function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA, SSI_angles] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold)
=======
function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA, viz_data] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold)
>>>>>>> Stashed changes

if isempty(eeg_data)
    error('Cannot process empty data');
end
if ~ismatrix(eeg_data)
    error('Input EEG data must be a 2D matrix (channels x samples).');
end
% pnts = size(eeg_data, 2); % Redundant
N_EEG_electrodes = size(eeg_data, 1);
% eeg_data = double(eeg_data); % REMOVED forced double cast
if ~isa(eeg_data, 'double') && ~isa(eeg_data, 'single')
    eeg_data = double(eeg_data); % Only cast if not already float
end

% Ensure refCOV matches precision of eeg_data
refCOV = cast(refCOV, 'like', eeg_data);

% Default signal_type if not provided
if nargin < 9 || isempty(signal_type)
    signal_type = 'eeg'; 
end

% Default minThreshold if not provided
if nargin < 10 || isempty(minThreshold)
    minThreshold = 0;
end

% Default maxThreshold if not provided
if nargin < 11 || isempty(maxThreshold)
    maxThreshold = 12;
end

%% Pad and Epoch Data
pnts_original = size(eeg_data, 2); 
epoch_samples = srate * epoch_size;

remainder = rem(pnts_original, epoch_samples);
if remainder ~= 0
    samples_to_pad = epoch_samples - remainder;
    reflection_segment = eeg_data(:, end-samples_to_pad+1:end);
    padding = fliplr(reflection_segment); % Flip the segment left-to-right
    eeg_data = [eeg_data, padding];
    % disp(['Data padded with ', num2str(samples_to_pad/srate, '%.2f'), ' seconds of reflected data.']);
end

% Epoch data stream 1
EEGdata_epoched = reshape(eeg_data, N_EEG_electrodes, epoch_samples, []);

% Epoch data stream 2 (shifted by half epoch)
shifting = epoch_samples / 2; 
eeg_data_2 = eeg_data(:, (shifting+1):(end-shifting));
EEGdata_epoched_2 = reshape(eeg_data_2, N_EEG_electrodes, epoch_samples, []);
[~,~,N_epochs] = size(EEGdata_epoched);
%% Calculate Covariance Matrix per Epoch
COV = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
COV_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
for epo=1:N_epochs-1
    COV(:,:,epo) = cov(EEGdata_epoched(:,:,epo)');
    COV_2(:,:,epo) = cov(EEGdata_epoched_2(:,:,epo)');
end
COV(:,:,N_epochs) = cov(EEGdata_epoched(:,:,N_epochs)');
%% Generalized Eigendecomposition (GEVD)
regularization_lambda = 0.05;
reg_val = trace(refCOV) / N_EEG_electrodes;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(N_EEG_electrodes, 'like', refCOV);
Evec = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
Eval = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
Evec_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
Eval_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
for i=1:N_epochs-1
    [Evec(:,:,i), Eval(:,:,i)] = eig(COV(:,:,i), refCOV_reg, 'chol');
    [Evec_2(:,:,i), Eval_2(:,:,i)] = eig(COV_2(:,:,i), refCOV_reg, 'chol');
end
[Evec(:,:,N_epochs), Eval(:,:,N_epochs)] = eig(COV(:,:,N_epochs), refCOV_reg, 'chol');


%% Determine Artifact Threshold and Clean EEG
%% Determine Noise Multiplier and Optimization Parameters
if ischar(artifact_threshold_type) && startsWith(artifact_threshold_type, 'auto')
    if strcmp(artifact_threshold_type,'auto++'), noise_multiplier = 0.5; % Super Aggressive
    elseif strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1.5; % Aggressive
    elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3;  % Balanced
    elseif strcmp(artifact_threshold_type,'auto-'), noise_multiplier = 6;  % Conservative
    elseif strcmp(artifact_threshold_type,'auto--'), noise_multiplier = 10; % Super Conservative
    else, noise_multiplier = 3; 
    end
else
    % Numeric input defines noise_multiplier linkage
    if isnumeric(artifact_threshold_type)
        val = artifact_threshold_type;
    else
        val = str2double(artifact_threshold_type);
    end
    noise_multiplier = 10 - val;
end

% Ensure valid multiplier
if isnan(noise_multiplier), noise_multiplier = 3; end

% --- Run SENSAI Optimization ---

% Extract reference covariance flattened vector for spatial correlation
tri_idx = triu(true(N_EEG_electrodes), 0);
refCOV_triu = refCOV(tri_idx);


% --- Optimization Method Switch ---
switch optimization_type
    case 'parabolic'
        [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Eval, Evec, noise_multiplier, COV, refCOV_triu, signal_type);
    
    case 'grid' % Restored grid search functionality
        automatic_thresholding_step_size = 1/3;
        AutomaticThresholdSweep = minThreshold:automatic_thresholding_step_size:maxThreshold;
        
        SIGNAL_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
        NOISE_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
        SENSAI_score = zeros(1, length(AutomaticThresholdSweep));
        if parallel
            parfor threshold_index=1:length(AutomaticThresholdSweep)
                artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                % Call SENSAI function
                [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval, Evec, noise_multiplier, COV, refCOV_triu, signal_type);
            end
        else
            for threshold_index=1:length(AutomaticThresholdSweep)
                artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                % Call SENSAI function
                [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval, Evec, noise_multiplier, COV, refCOV_triu, signal_type);
            end
        end
        [~, SENSAI_index] = max(SENSAI_score);
        NOISE_changepoint_index = findchangepts(diff(smoothdata(NOISE_subspace_similarity, "movmean",6)),Statistic="mean", MaxNumChanges=2);
    
        if isempty(NOISE_changepoint_index)
            NOISE_changepoint_index = length(AutomaticThresholdSweep);      
        end
        if SENSAI_index > NOISE_changepoint_index(1)
            optimal_artifact_threshold = AutomaticThresholdSweep(NOISE_changepoint_index(1));
        else
            optimal_artifact_threshold = AutomaticThresholdSweep(SENSAI_index);
        end
end

artifact_threshold = optimal_artifact_threshold;
% Pre-calculate cosine weights for efficiency
cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);

[cleaned_data_1, artifacts_data_1, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec, cosine_weights, signal_type);
[cleaned_data_2, artifacts_data_2, ~] = clean_EEG(EEGdata_epoched_2, srate, epoch_size, artifact_threshold, refCOV, Eval_2, Evec_2, cosine_weights, signal_type);

% Clear Stream 2 inputs as they are no longer needed
clear EEGdata_epoched_2 Evec_2 Eval_2 COV_2;

%% Combine the two processed streams using cosine weighting
% cosine_weights is already calculated

size_reconstructed_2 = size(cleaned_data_2, 2);
sample_end = size_reconstructed_2 - shifting;
% Apply weights to the second (shifted) stream
cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
cleaned_data_2(:, sample_end+1:end) = cleaned_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
artifacts_data_2(:, 1:shifting) = artifacts_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
artifacts_data_2(:, sample_end+1:end) = artifacts_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);

% Combine streams (Optimize memory by clearing variables)
cleaned_data = cleaned_data_1;
clear cleaned_data_1; % Release memory

artifacts_data = artifacts_data_1;
clear artifacts_data_1; % Release memory

cleaned_data(:, shifting+1:shifting+size_reconstructed_2) = cleaned_data(:, shifting+1:shifting+size_reconstructed_2) + cleaned_data_2;
clear cleaned_data_2; % Release memory

artifacts_data(:, shifting+1:shifting+size_reconstructed_2) = artifacts_data(:, shifting+1:shifting+size_reconstructed_2) + artifacts_data_2;
clear artifacts_data_2; % Release memory

% Remove padding to restore original data length
cleaned_data = cleaned_data(:, 1:pnts_original);
artifacts_data = artifacts_data(:, 1:pnts_original);

%% Calculate final SENSAI score
[~, ~, SENSAI_score] = SENSAI(artifact_threshold_out, refCOV, Eval, Evec, noise_multiplier, COV, refCOV_triu, signal_type);

% Calculate mean ENOVA for this band (average of per-epoch variance ratios)
original_data = cleaned_data + artifacts_data;

% Reshape into epochs (channels x samples x epochs)
epoch_samples = srate * epoch_size;
% Handle potential padding/truncation: use floor to get full epochs
num_epochs_possible = floor(size(original_data, 2) / epoch_samples);
len_to_use = num_epochs_possible * epoch_samples;

original_epoched = reshape(original_data(:, 1:len_to_use), size(original_data, 1), epoch_samples, []);
artifacts_epoched = reshape(artifacts_data(:, 1:len_to_use), size(artifacts_data, 1), epoch_samples, []);

num_epochs = size(original_epoched, 3);
enova_per_epoch = zeros(1, num_epochs);

for i = 1:num_epochs
    % Calculate variance for this epoch (across all channels and time points in epoch)
    var_orig = var(reshape(original_epoched(:,:,i), [], 1));
    var_art = var(reshape(artifacts_epoched(:,:,i), [], 1));
    
    if var_orig > 0
        enova_per_epoch(i) = var_art / var_orig;
    else
        enova_per_epoch(i) = 0; % Avoid division by zero
    end
end

if num_epochs > 0
    ENOVA = mean(enova_per_epoch);
else
    ENOVA = 0;
end

%% SSI components for visualization (Optional output)
if nargout > 5
<<<<<<< Updated upstream
    SSI_top_PCs = 3;
    num_channels = size(refCOV, 1);
    if SSI_top_PCs > num_channels, SSI_top_PCs = num_channels; end
=======
    num_channels = size(refCOV, 1);
>>>>>>> Stashed changes
    
    [Vref, Dref] = eig(refCOV);
    [~, idx] = sort(diag(Dref), 'descend');
    basis_ref = Vref(:, idx(1:SSI_top_PCs));
    
    cleaned_epoched = reshape(cleaned_data(:, 1:len_to_use), size(cleaned_data, 1), epoch_samples, []);
    
<<<<<<< Updated upstream
    num_epochs = size(cleaned_epoched, 3);
    angs_before = zeros(num_epochs, SSI_top_PCs);
    angs_after = zeros(num_epochs, SSI_top_PCs);
    angs_artifacts = zeros(num_epochs, SSI_top_PCs);
=======
    num_epochs = size(original_epoched, 3);
    ssi_before = zeros(num_epochs, 1);
    lpow_before = zeros(num_epochs, 1);
    ssi_after = zeros(num_epochs, 1);
    lpow_after = zeros(num_epochs, 1);
    ssi_artifacts = zeros(num_epochs, 1);
    lpow_artifacts = zeros(num_epochs, 1);
>>>>>>> Stashed changes
    
    for epo = 1:num_epochs
        % Before
        C_before = cov(original_epoched(:,:,epo)');
<<<<<<< Updated upstream
        angs_before(epo, :) = extract_single_angles_inline(C_before, basis_ref, SSI_top_PCs);
        % After
        C_after = cov(cleaned_epoched(:,:,epo)');
        angs_after(epo, :) = extract_single_angles_inline(C_after, basis_ref, SSI_top_PCs);
        % Artifacts
        C_artifacts = cov(artifacts_epoched(:,:,epo)');
        angs_artifacts(epo, :) = extract_single_angles_inline(C_artifacts, basis_ref, SSI_top_PCs);
    end
    SSI_angles.angs_before = angs_before;
    SSI_angles.angs_after = angs_after;
    SSI_angles.angs_artifacts = angs_artifacts;
=======
        angs_before = subspace_angles(basis_ref, basis_vis_inline(C_before, SSI_top_PCs));
        ssi_before(epo) = prod(angs_before) ^ (1/SSI_top_PCs);
        lpow_before(epo) = 10 * log10(trace(C_before));
        
        % After
        C_after = cov(cleaned_epoched(:,:,epo)');
        angs_after = subspace_angles(basis_ref, basis_vis_inline(C_after, SSI_top_PCs));
        ssi_after(epo) = prod(angs_after) ^ (1/SSI_top_PCs);
        lpow_after(epo) = 10 * log10(trace(C_after));
        
    % Artifacts
        pow_art = trace(cov(artifacts_epoched(:,:,epo)'));
        pow_artifacts(epo) = pow_art;
        if pow_art > eps
            C_artifacts = cov(artifacts_epoched(:,:,epo)');
            angs_artifacts = subspace_angles(basis_ref, basis_vis_inline(C_artifacts, SSI_top_PCs));
            ssi_artifacts(epo) = prod(angs_artifacts) ^ (1/SSI_top_PCs);
        end
    end
    
    % Only track valid artifact epochs (where power > 0)
    valid_art = pow_artifacts > eps;
    ssi_artifacts_valid = ssi_artifacts(valid_art);
    lpow_artifacts_valid = 10 * log10(pow_artifacts(valid_art));

    % Calculate Visual Silhouette (using min-max normalized axes to match visual plot spread)
    if isempty(lpow_artifacts_valid)
        viz_data.silhouette = 0;
    else
        p_min = min([lpow_after(:); lpow_artifacts_valid(:); 0]); 
        p_max = max([lpow_after(:); lpow_artifacts_valid(:); 1]);
        p_range = max(1e-6, p_max - p_min);
        X_s = [ssi_after(:), (lpow_after(:) - p_min) / p_range];
        X_n = [ssi_artifacts_valid(:), (lpow_artifacts_valid(:) - p_min) / p_range];
        
        % a_i: Internal Cohesion
        D_sig = sqrt(max(0, bsxfun(@plus, sum(X_s.^2, 2), sum(X_s.^2, 2)') - 2*(X_s*X_s')));
        if num_epochs > 1
            a_i_vis = (sum(D_sig, 2)) ./ (num_epochs - 1);
        else
            a_i_vis = zeros(num_epochs, 1);
        end
        
        % b_i: Standard Separation (Mean distance)
        D_bg = sqrt(max(0, bsxfun(@plus, sum(X_s.^2, 2), sum(X_n.^2, 2)') - 2*(X_s*X_n')));
        b_i_vis = mean(D_bg, 2);
        
        s_i_vis = (b_i_vis - a_i_vis) ./ max(a_i_vis, b_i_vis);
        s_i_vis(isnan(s_i_vis)) = 0;
        
        viz_data.silhouette = mean(s_i_vis);
    end
    
    viz_data.ssi_before = ssi_before;
    viz_data.lpow_before = lpow_before;
    viz_data.ssi_after = ssi_after;
    viz_data.lpow_after = lpow_after;
    viz_data.ssi_artifacts = ssi_artifacts_valid;
    viz_data.lpow_artifacts = lpow_artifacts_valid;
    viz_data.sensai_score = SENSAI_score;
>>>>>>> Stashed changes
end

end

<<<<<<< Updated upstream
function angs = extract_single_angles_inline(C, basis_ref, top_PCs)
    if all(C(:) == 0) || any(isnan(C(:)))
        angs = zeros(1, top_PCs);
=======
function basis = basis_vis_inline(C, top_PCs)
    if all(C(:) == 0) || any(isnan(C(:)))
        basis = zeros(size(C,1), top_PCs);
>>>>>>> Stashed changes
        return;
    end
    [V, D] = eig(C);
    [~, idx] = sort(diag(D), 'descend');
<<<<<<< Updated upstream
    basis_c = V(:, idx(1:top_PCs));
    cos_theta = subspace_angles(basis_c, basis_ref);
    angs = cos_theta(:)';
=======
    basis = V(:, idx(1:top_PCs));
>>>>>>> Stashed changes
end