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

function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallelize, signal_type, minThreshold, maxThreshold, smoothing_window_seconds)

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
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;

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

% Default smoothing_window_seconds if not provided
if nargin < 12 || isempty(smoothing_window_seconds)
    smoothing_window_seconds = Inf;
end

%% Pad and Epoch Data
pnts_original = size(eeg_data, 2); 
epoch_samples = round(srate * epoch_size);

remainder = rem(pnts_original, epoch_samples);
if remainder ~= 0
    samples_to_pad = epoch_samples - remainder;
    padding = local_reflect_pad(eeg_data, samples_to_pad);
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
if ~isinf(smoothing_window_seconds)
    % =========================================================================
    % MEMORY-OPTIMIZED SLIDING WINDOW PATH
    % =========================================================================
    disp('Executing Sliding Window GEDAI Denoising...');
    
    window_seconds = smoothing_window_seconds;
    window_epochs = max(1, round(window_seconds / epoch_size));
    step_epochs = max(1, round(window_epochs / 2));
    
    num_windows = max(1, ceil((N_epochs - window_epochs) / step_epochs) + 1);
    if N_epochs <= window_epochs
        num_windows = 1;
        window_epochs = N_epochs;
    end
    
    window_centers = zeros(1, num_windows);
    optimal_threshold_per_window = zeros(1, num_windows);
    
    regularization_lambda = 0.05;
    reg_val = trace(refCOV) / N_EEG_electrodes;
    refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(N_EEG_electrodes, 'like', refCOV);
    refCOV_reg = (refCOV_reg + refCOV_reg') / 2;
    
    % --- Determine Noise Multiplier and Optimization Parameters ---
    is_auto_threshold = (ischar(artifact_threshold_type) || isstring(artifact_threshold_type)) && startsWith(string(artifact_threshold_type), "auto");
    auto_scale = 0.5;
    if is_auto_threshold
        if strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1; auto_scale = 1/3;
        elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3; auto_scale = 0.5;
        elseif strcmp(artifact_threshold_type,'auto-'), noise_multiplier = 3; auto_scale = 1.0;
        elseif strncmp(char(artifact_threshold_type),'auto:',5), auto_scale = local_strength_to_sc(str2double(extractAfter(string(artifact_threshold_type),"auto:"))); noise_multiplier = 3 * auto_scale; else, noise_multiplier = 3; auto_scale = 0.5;
        end
    else
        if isnumeric(artifact_threshold_type)
            val = artifact_threshold_type;
        else
            val = str2double(artifact_threshold_type);
        end
        noise_multiplier = 10 - val;
    end
    if isnan(noise_multiplier), noise_multiplier = 3; end
    
    % Pre-calculate RefCOV eigenvectors for SENSAI
    if strcmpi(signal_type, 'eeg')
        SSI_top_PCs = min(3, N_EEG_electrodes);
    elseif strcmpi(signal_type, 'meg')
        % Option 12: Adaptive GEVD Artifact Spectrum Pre-Scan
        % Senses whether artifact is focal rank-1/2 (k = 2) or multi-dipole/complex (k = 3)
        cov_data_prescan = cov(double(eeg_data)');
        cov_data_prescan = (cov_data_prescan + cov_data_prescan') / 2;
        [~, D_gevd_prescan] = eig(cov_data_prescan, refCOV_reg);
        evals_prescan = sort(diag(D_gevd_prescan), 'descend');
        
        med_val_prescan = median(evals_prescan);
        if med_val_prescan > 0
            norm_evals = evals_prescan / med_val_prescan;
        else
            norm_evals = evals_prescan;
        end
        
        % Check if component 3 has elevated power above baseline noise floor,
        % or if components 1 & 2 indicate broad multi-dipole distribution
        if length(norm_evals) >= 3 && (norm_evals(3) > 200 || (norm_evals(1) > 0 && norm_evals(2) / norm_evals(1) > 0.20 && norm_evals(1) < 12000))
            SSI_top_PCs = min(3, N_EEG_electrodes);
        else
            SSI_top_PCs = min(2, N_EEG_electrodes);
        end
    end
    
    [Vs, Ds] = eig(refCOV_reg);
    [~, sidxS_Template_cov] = sort(diag(Ds), 'descend');
    evecs_Template_cov = Vs(:, sidxS_Template_cov(1:SSI_top_PCs));
    
    % Loop over local windows to optimize threshold curves
    for w = 1:num_windows
        idx_start = (w - 1) * step_epochs + 1;
        idx_end = min(N_epochs, idx_start + window_epochs - 1);
        if w == num_windows && (idx_end - idx_start + 1) < window_epochs/2 && num_windows > 1
            idx_start = max(1, N_epochs - window_epochs + 1);
            idx_end = N_epochs;
        end
        window_centers(w) = (idx_start + idx_end) / 2;
        
        w_len = idx_end - idx_start + 1;
        start_sample = (idx_start - 1) * epoch_samples + 1;
        end_sample = idx_end * epoch_samples;
        
        eeg_data_sub = eeg_data(:, start_sample:end_sample);
        EEGdata_epoched_sub = reshape(eeg_data_sub, N_EEG_electrodes, epoch_samples, w_len);
        
        COV_sub = zeros(N_EEG_electrodes, N_EEG_electrodes, w_len, 'like', eeg_data);
        for epo = 1:w_len
            COV_sub(:,:,epo) = cov(EEGdata_epoched_sub(:,:,epo)');
        end
        
        Evec_sub = zeros(N_EEG_electrodes, N_EEG_electrodes, w_len, 'like', eeg_data);
        Eval_sub = zeros(N_EEG_electrodes, N_EEG_electrodes, w_len, 'like', eeg_data);
        for i = 1:w_len
            c_ep = (COV_sub(:,:,i) + COV_sub(:,:,i)') / 2;
            c_tr = trace(c_ep) / N_EEG_electrodes;
            if c_tr > 0
                c_ep = (1 - 1e-9) * c_ep + (1e-9 * c_tr) * refCOV_reg;
            end
            COV_sub(:,:,i) = c_ep;
            [Evec_sub(:,:,i), Eval_sub(:,:,i)] = eig(c_ep, refCOV_reg, 'chol');
        end
        
        if is_auto_threshold && (strcmpi(signal_type, 'eeg') || strcmpi(signal_type, 'meg'))
            noise_multiplier = auto_scale * local_compute_z3_noise_multiplier( ...
                Eval_sub, Evec_sub, refCOV_reg, evecs_Template_cov, SSI_top_PCs, minThreshold, epoch_samples);
        end
        
        switch optimization_type
            case 'parabolic'
                [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Eval_sub, Evec_sub, noise_multiplier, COV_sub, evecs_Template_cov, signal_type, SSI_top_PCs);
            case 'grid'
                automatic_thresholding_step_size = 1/3;
                AutomaticThresholdSweep = minThreshold:automatic_thresholding_step_size:maxThreshold;
                SIGNAL_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
                NOISE_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
                SENSAI_score_sweep = zeros(1, length(AutomaticThresholdSweep));
                if parallelize
                    parfor threshold_index=1:length(AutomaticThresholdSweep)
                        artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                        [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score_sweep(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval_sub, Evec_sub, noise_multiplier, COV_sub, evecs_Template_cov, signal_type, SSI_top_PCs);
                    end
                else
                    for threshold_index=1:length(AutomaticThresholdSweep)
                        artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                        [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score_sweep(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval_sub, Evec_sub, noise_multiplier, COV_sub, evecs_Template_cov, signal_type, SSI_top_PCs);
                    end
                end
                [~, SENSAI_index] = max(SENSAI_score_sweep);
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
        optimal_threshold_per_window(w) = optimal_artifact_threshold;
        
        clear eeg_data_sub EEGdata_epoched_sub COV_sub Evec_sub Eval_sub;
    end
    
    if num_windows > 1
        if num_windows >= 3
            optimal_threshold_per_window = smoothdata(optimal_threshold_per_window, 'movmean', 3);
        end
        padded_centers    = [1, window_centers, N_epochs];
        padded_thresholds = [optimal_threshold_per_window(1), optimal_threshold_per_window, optimal_threshold_per_window(end)];
        [unique_centers, unique_idx] = unique(padded_centers);
        unique_thresholds = padded_thresholds(unique_idx);
        artifact_threshold_array = interp1(unique_centers, unique_thresholds, 1:N_epochs, 'makima');
    else
        artifact_threshold_array = repmat(optimal_threshold_per_window, 1, N_epochs);
    end
    artifact_threshold_array = max(minThreshold, min(maxThreshold, artifact_threshold_array));
    
    artifact_threshold = artifact_threshold_array;
    cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);
    
    artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
    if isempty(artifact_threshold_2)
        artifact_threshold_2 = artifact_threshold;
    end
    
    % Pre-allocate outputs for chunked execution
    cleaned_data_1 = zeros(N_EEG_electrodes, N_epochs * epoch_samples, 'like', eeg_data);
    artifacts_data_1 = zeros(N_EEG_electrodes, N_epochs * epoch_samples, 'like', eeg_data);
    cleaned_data_2 = zeros(N_EEG_electrodes, (N_epochs - 1) * epoch_samples, 'like', eeg_data);
    artifacts_data_2 = zeros(N_EEG_electrodes, (N_epochs - 1) * epoch_samples, 'like', eeg_data);
    
    SIGNAL_subspace_dist = zeros(1, N_epochs);
    NOISE_subspace_dist = zeros(1, N_epochs);
    Template_guess = evecs_Template_cov(:, 1:min(size(evecs_Template_cov, 2), SSI_top_PCs));
    mean_thresh = mean(artifact_threshold_array);
    
    % Adapt chunk size to both epoch duration and channel-count-driven memory pressure.
    chunk_size = local_chunk_size(N_EEG_electrodes, epoch_samples, epoch_size, eeg_data);
    num_chunks = ceil(N_epochs / chunk_size);
    for chunk = 1:num_chunks
        c_start = (chunk - 1) * chunk_size + 1;
        c_end = min(N_epochs, chunk * chunk_size);
        c_len = c_end - c_start + 1;
        
        chunk_samples_start = (c_start - 1) * epoch_samples + 1;
        chunk_samples_end = c_end * epoch_samples;
        eeg_data_chunk = eeg_data(:, chunk_samples_start:chunk_samples_end);
        
        EEGdata_epoched_chunk = reshape(eeg_data_chunk, N_EEG_electrodes, epoch_samples, c_len);
        
        Evec_chunk = zeros(N_EEG_electrodes, N_EEG_electrodes, c_len, 'like', eeg_data);
        Eval_chunk = zeros(N_EEG_electrodes, N_EEG_electrodes, c_len, 'like', eeg_data);
        for i = 1:c_len
            cov_epoch = cov(EEGdata_epoched_chunk(:,:,i)');
            cov_epoch = (cov_epoch + cov_epoch') / 2;
            c_tr = trace(cov_epoch) / N_EEG_electrodes;
            if c_tr > 0
                cov_epoch = (1 - 1e-9) * cov_epoch + (1e-9 * c_tr) * refCOV_reg;
            end
            [Evec_chunk(:,:,i), Eval_chunk(:,:,i)] = eig(cov_epoch, refCOV_reg, 'chol');
        end
        
        chunk_threshold = artifact_threshold_array(c_start:c_end);
        [cleaned_chunk, artifacts_chunk, artifact_threshold_out] = clean_EEG(EEGdata_epoched_chunk, srate, epoch_size, chunk_threshold, refCOV, Eval_chunk, Evec_chunk, cosine_weights, signal_type, c_start, N_epochs, minThreshold);
        
        cleaned_data_1(:, chunk_samples_start:chunk_samples_end) = cleaned_chunk;
        artifacts_data_1(:, chunk_samples_start:chunk_samples_end) = artifacts_chunk;
        
        % Accumulate analytical SENSAI statistics locally
        % Fast Hybrid Associative Double-QR for SENSAI statistics tracker
        base_diag = (1 : (N_EEG_electrodes + 1) : N_EEG_electrodes^2)';
        all_indices = base_diag + (0 : c_len-1) * N_EEG_electrodes^2;
        all_diagonals = Eval_chunk(all_indices(:));
        magnitudes = abs(all_diagonals);
        all_evals_chunk = reshape(magnitudes, N_EEG_electrodes, c_len);
        max_mag_chunk = max(magnitudes);
        if max_mag_chunk > 0
            valid_mags_chunk = max(magnitudes, max(1e-12 * max_mag_chunk, 1e-30));
        else
            valid_mags_chunk = magnitudes(magnitudes > 0);
        end
        log_Eig_val_chunk = log(valid_mags_chunk) + 100;
        
        correction_factor = 1.00;
        T1 = correction_factor * (105 - mean_thresh) / 100;
        if strcmpi(signal_type, 'eeg')
            percentile_threshold = 98;
        elseif strcmpi(signal_type, 'meg')
            percentile_threshold = 99;
        end
        if ~isempty(log_Eig_val_chunk)
            Treshold_chunk = T1 * prctile(log_Eig_val_chunk, percentile_threshold);
        else
            Treshold_chunk = 0;
        end
        threshold_val = exp(Treshold_chunk - 100);
        
        T_proj = refCOV_reg * Template_guess;
        M_proj = size(Template_guess, 2);
        tol_val = 1e-30;
        
        for i = 1:c_len
            global_epo_idx = c_start + i - 1;
            current_evals = all_evals_chunk(:, i);
            active_mask = current_evals > max(1e-7 * max(current_evals), tol_val);
            bad_indices = (current_evals >= threshold_val) & active_mask;
            num_bad = sum(bad_indices);
            good_indices = (~bad_indices) & active_mask;
            num_good = sum(good_indices);
            
            % --- SIGNAL SUBSPACE ---
            if num_good >= M_proj
                % Fast associative path
                Evec_good = Evec_chunk(:, good_indices, i);
                d_good = current_evals(good_indices);
                Y1_sig = refCOV_reg * (Evec_good * (d_good .* (Evec_good' * T_proj)));
                [Q1_sig, ~] = qr(Y1_sig, 0);
                T_sig = refCOV_reg * Q1_sig;
                Y2_sig = refCOV_reg * (Evec_good * (d_good .* (Evec_good' * T_sig)));
                [evecs_signal, ~] = qr(Y2_sig, 0);
                SIGNAL_subspace_dist(global_epo_idx) = abs(det(evecs_signal' * Template_guess));
            else
                SIGNAL_subspace_dist(global_epo_idx) = 0;
            end
            
            % --- NOISE SUBSPACE ---
            if strcmpi(signal_type, 'meg')
                % Energy-Fraction Rank (Optimal for MEG)
                if num_bad > 0
                    Evec_bad = Evec_chunk(:, bad_indices, i);
                    P_bad = refCOV_reg * Evec_bad;
                    [Q_bad, ~] = qr(P_bad, 0);
                    s = svd(Q_bad' * Template_guess);
                    NOISE_subspace_dist(global_epo_idx) = sum(s.^6);
                else
                    NOISE_subspace_dist(global_epo_idx) = 0;
                end
            else
                % Subspace Volume Determinant + Rank-1 Dipolar Rhythm Guardrail (Optimal for EEG)
                if num_bad >= M_proj
                    % Fast associative path
                    Evec_bad = Evec_chunk(:, bad_indices, i);
                    d_bad = current_evals(bad_indices);
                    Y1_noise = refCOV_reg * (Evec_bad * (d_bad .* (Evec_bad' * T_proj)));
                    [Q1_noise, ~] = qr(Y1_noise, 0);
                    T_noise = refCOV_reg * Q1_noise;
                    Y2_noise = refCOV_reg * (Evec_bad * (d_bad .* (Evec_bad' * T_noise)));
                    [evecs_noise, ~] = qr(Y2_noise, 0);
                    nsi_det = abs(det(evecs_noise' * Template_guess));
                    P_bad = refCOV_reg * Evec_bad;
                    [Q_bad, ~] = qr(P_bad, 0);
                    s_bad = svd(Q_bad' * Template_guess);
                    NOISE_subspace_dist(global_epo_idx) = 0.88 * nsi_det + 0.12 * sum(s_bad.^6);
                elseif num_bad > 0
                    Evec_bad = Evec_chunk(:, bad_indices, i);
                    P_bad = refCOV_reg * Evec_bad;
                    [Q_bad, ~] = qr(P_bad, 0);
                    s_bad = svd(Q_bad' * Template_guess);
                    NOISE_subspace_dist(global_epo_idx) = 0.12 * sum(s_bad.^6);
                else
                    evecs_noise = eye(N_EEG_electrodes, M_proj);
                    NOISE_subspace_dist(global_epo_idx) = abs(det(evecs_noise' * Template_guess));
                end
            end
        end
        
        clear eeg_data_chunk EEGdata_epoched_chunk Evec_chunk Eval_chunk cleaned_chunk artifacts_chunk;
    end
    
    % Clean Shifted Stream 2 in Chunks
    num_chunks_2 = ceil((N_epochs - 1) / chunk_size);
    for chunk = 1:num_chunks_2
        c_start = (chunk - 1) * chunk_size + 1;
        c_end = min(N_epochs - 1, chunk * chunk_size);
        c_len = c_end - c_start + 1;
        
        chunk_samples_start = (c_start - 1) * epoch_samples + 1;
        chunk_samples_end = c_end * epoch_samples;
        eeg_data_chunk = eeg_data_2(:, chunk_samples_start:chunk_samples_end);
        
        EEGdata_epoched_chunk = reshape(eeg_data_chunk, N_EEG_electrodes, epoch_samples, c_len);
        
        Evec_chunk = zeros(N_EEG_electrodes, N_EEG_electrodes, c_len, 'like', eeg_data);
        Eval_chunk = zeros(N_EEG_electrodes, N_EEG_electrodes, c_len, 'like', eeg_data);
        for i = 1:c_len
            cov_epoch = cov(EEGdata_epoched_chunk(:,:,i)');
            cov_epoch = (cov_epoch + cov_epoch') / 2;
            c_tr = trace(cov_epoch) / N_EEG_electrodes;
            if c_tr > 0
                cov_epoch = (1 - 1e-9) * cov_epoch + (1e-9 * c_tr) * refCOV_reg;
            end
            [Evec_chunk(:,:,i), Eval_chunk(:,:,i)] = eig(cov_epoch, refCOV_reg, 'chol');
        end
        
        chunk_threshold = artifact_threshold_2(c_start:c_end);
        [cleaned_chunk, artifacts_chunk] = clean_EEG(EEGdata_epoched_chunk, srate, epoch_size, chunk_threshold, refCOV, Eval_chunk, Evec_chunk, cosine_weights, signal_type, c_start, N_epochs - 1, minThreshold);
        
        cleaned_data_2(:, chunk_samples_start:chunk_samples_end) = cleaned_chunk;
        artifacts_data_2(:, chunk_samples_start:chunk_samples_end) = artifacts_chunk;
        
        clear eeg_data_chunk EEGdata_epoched_chunk Evec_chunk Eval_chunk cleaned_chunk artifacts_chunk;
    end
    
    % Reconstruct overlapping segments
    size_reconstructed_2 = size(cleaned_data_2, 2);
    if size_reconstructed_2 > 0
        sample_end = size_reconstructed_2 - shifting;
        cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
        cleaned_data_2(:, sample_end+1:end) = cleaned_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
        artifacts_data_2(:, 1:shifting) = artifacts_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
        artifacts_data_2(:, sample_end+1:end) = artifacts_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
    end
    
    cleaned_data = cleaned_data_1;
    artifacts_data = artifacts_data_1;
    clear cleaned_data_1 artifacts_data_1;
    
    if size_reconstructed_2 > 0
        cleaned_data(:, shifting+1:shifting+size_reconstructed_2) = cleaned_data(:, shifting+1:shifting+size_reconstructed_2) + cleaned_data_2;
        artifacts_data(:, shifting+1:shifting+size_reconstructed_2) = artifacts_data(:, shifting+1:shifting+size_reconstructed_2) + artifacts_data_2;
    end
    clear cleaned_data_2 artifacts_data_2;
    
    cleaned_data = cleaned_data(:, 1:pnts_original);
    artifacts_data = artifacts_data(:, 1:pnts_original);
    
    % Compute final SENSAI score from the tracked vectors
    SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_dist);
    NOISE_subspace_similarity = 100 * mean(NOISE_subspace_dist);
    SENSAI_score = SIGNAL_subspace_similarity - (noise_multiplier * NOISE_subspace_similarity);
    
    artifact_threshold_out = artifact_threshold;
    
    % Analytical ENOVA calculation
    original_data = cleaned_data + artifacts_data;
    num_epochs_possible = floor(size(original_data, 2) / epoch_samples);
    len_to_use = num_epochs_possible * epoch_samples;
    original_epoched = reshape(original_data(:, 1:len_to_use), size(original_data, 1), epoch_samples, []);
    artifacts_epoched = reshape(artifacts_data(:, 1:len_to_use), size(artifacts_data, 1), epoch_samples, []);
    num_epochs_enova = size(original_epoched, 3);
    if num_epochs_enova > 0
        original_flat = reshape(original_epoched, [], num_epochs_enova);
        artifacts_flat = reshape(artifacts_epoched, [], num_epochs_enova);
        var_orig = var(original_flat, 0, 1);
        var_art = var(artifacts_flat, 0, 1);
        enova_per_epoch = zeros(1, num_epochs_enova);
        valid = var_orig > 0;
        enova_per_epoch(valid) = var_art(valid) ./ var_orig(valid);
        ENOVA = mean(enova_per_epoch);
    else
        ENOVA = 0;
    end
    
else
    % =========================================================================
    % ORIGINAL GLOBAL PATH (Fully backwards-compatible, unchanged)
    % =========================================================================
    
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
    refCOV_reg = (refCOV_reg + refCOV_reg') / 2;
    Evec = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
    Eval = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs, 'like', eeg_data);
    Evec_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
    Eval_2 = zeros(N_EEG_electrodes, N_EEG_electrodes, N_epochs-1, 'like', eeg_data);
    for i=1:N_epochs-1
        COV(:,:,i) = (COV(:,:,i) + COV(:,:,i)') / 2;
        [Evec(:,:,i), Eval(:,:,i)] = eig(COV(:,:,i), refCOV_reg, 'chol');
        COV_2(:,:,i) = (COV_2(:,:,i) + COV_2(:,:,i)') / 2;
        [Evec_2(:,:,i), Eval_2(:,:,i)] = eig(COV_2(:,:,i), refCOV_reg, 'chol');
    end
    COV(:,:,N_epochs) = (COV(:,:,N_epochs) + COV(:,:,N_epochs)') / 2;
    [Evec(:,:,N_epochs), Eval(:,:,N_epochs)] = eig(COV(:,:,N_epochs), refCOV_reg, 'chol');
    
    
    %% Determine Artifact Threshold and Clean EEG
    %% Determine Noise Multiplier and Optimization Parameters
    is_auto_threshold = (ischar(artifact_threshold_type) || isstring(artifact_threshold_type)) && startsWith(string(artifact_threshold_type), "auto");
    auto_scale = 0.5;
    if is_auto_threshold
        if strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1; auto_scale = 1/3;
        elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3; auto_scale = 0.5;
        elseif strcmp(artifact_threshold_type,'auto-'), noise_multiplier = 3; auto_scale = 1.0;
        elseif strncmp(char(artifact_threshold_type),'auto:',5), auto_scale = local_strength_to_sc(str2double(extractAfter(string(artifact_threshold_type),"auto:"))); noise_multiplier = 3 * auto_scale; else, noise_multiplier = 3; auto_scale = 0.5;
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
    
    % Pre-calculate RefCOV eigenvectors for SENSAI
    
    
    if strcmpi(signal_type, 'eeg')
         SSI_top_PCs = min(3, N_EEG_electrodes);
    
    elseif strcmpi(signal_type, 'meg')
        % Option 12: Adaptive GEVD Artifact Spectrum Pre-Scan
        % Senses whether artifact is focal rank-1/2 (k = 2) or multi-dipole/complex (k = 3)
        cov_data_prescan = cov(double(eeg_data)');
        cov_data_prescan = (cov_data_prescan + cov_data_prescan') / 2;
        [~, D_gevd_prescan] = eig(cov_data_prescan, refCOV_reg);
        evals_prescan = sort(diag(D_gevd_prescan), 'descend');
        
        med_val_prescan = median(evals_prescan);
        if med_val_prescan > 0
            norm_evals = evals_prescan / med_val_prescan;
        else
            norm_evals = evals_prescan;
        end
        
        % Check if component 3 has elevated power above baseline noise floor,
        % or if components 1 & 2 indicate broad multi-dipole distribution
        if length(norm_evals) >= 3 && (norm_evals(3) > 200 || (norm_evals(1) > 0 && norm_evals(2) / norm_evals(1) > 0.20 && norm_evals(1) < 12000))
            SSI_top_PCs = min(3, N_EEG_electrodes);
        else
            SSI_top_PCs = min(2, N_EEG_electrodes);
        end
    end
    
    % Apply SSI_top_PCs
    [Vs, Ds] = eig(refCOV_reg);
    [~, sidxS_Template_cov] = sort(diag(Ds), 'descend');
    evecs_Template_cov = Vs(:, sidxS_Template_cov(1:SSI_top_PCs));
    
    
    % --- Optimization Method Switch (Sliding Window) ---
    % smoothing_window_seconds controls the sliding window size.
    % Inf (default) means use the entire file as one window (no sliding = original global behavior).
    if nargin < 12 || isempty(smoothing_window_seconds)
        smoothing_window_seconds = Inf;
    end
    if isinf(smoothing_window_seconds)
        window_seconds = N_epochs * epoch_size; % effectively the whole file
    else
        window_seconds = smoothing_window_seconds;
    end
    window_epochs = max(1, round(window_seconds / epoch_size));
    step_epochs = max(1, round(window_epochs / 2));
    
    num_windows = max(1, ceil((N_epochs - window_epochs) / step_epochs) + 1);
    if N_epochs <= window_epochs
        num_windows = 1;
        window_epochs = N_epochs;
    end
    
    window_centers = zeros(1, num_windows);
    optimal_threshold_per_window = zeros(1, num_windows);
    
    for w = 1:num_windows
        idx_start = (w - 1) * step_epochs + 1;
        idx_end = min(N_epochs, idx_start + window_epochs - 1);
        
        % Ensure the last window is reasonably sized
        if w == num_windows && (idx_end - idx_start + 1) < window_epochs/2 && num_windows > 1
            idx_start = max(1, N_epochs - window_epochs + 1);
            idx_end = N_epochs;
        end
        
        window_centers(w) = (idx_start + idx_end) / 2;
        
        Eval_sub = Eval(:,:,idx_start:idx_end);
        Evec_sub = Evec(:,:,idx_start:idx_end);
        COV_sub = COV(:,:,idx_start:idx_end);
        
        if is_auto_threshold && (strcmpi(signal_type, 'eeg') || strcmpi(signal_type, 'meg'))
            noise_multiplier = auto_scale * local_compute_z3_noise_multiplier( ...
                Eval_sub, Evec_sub, refCOV_reg, evecs_Template_cov, SSI_top_PCs, minThreshold, epoch_samples);
        end
        
        switch optimization_type
            case 'parabolic'
                [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Eval_sub, Evec_sub, noise_multiplier, COV_sub, evecs_Template_cov, signal_type, SSI_top_PCs);
            
            case 'grid' % Restored grid search functionality
                automatic_thresholding_step_size = 1/3;
                AutomaticThresholdSweep = minThreshold:automatic_thresholding_step_size:maxThreshold;
                
                SIGNAL_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
                NOISE_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
                SENSAI_score = zeros(1, length(AutomaticThresholdSweep));
                if parallelize
                    parfor threshold_index=1:length(AutomaticThresholdSweep)
                        artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                        % Call SENSAI function
                        [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval_sub, Evec_sub, noise_multiplier, COV_sub, evecs_Template_cov, signal_type, SSI_top_PCs);
                    end
                else
                    for threshold_index=1:length(AutomaticThresholdSweep)
                        artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                        % Call SENSAI function
                        [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Eval_sub, Evec_sub, noise_multiplier, COV_sub, evecs_Template_cov, signal_type, SSI_top_PCs);
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
        optimal_threshold_per_window(w) = optimal_artifact_threshold;
    end
    
    if num_windows > 1
        % Smooth per-window thresholds before interpolation.
        % The SENSAI optimizer can return different values for adjacent windows
        % even when data hasn't changed much. A moving average over 3 windows
        % dampens these spurious jumps while preserving real trends.
        if num_windows >= 3
            optimal_threshold_per_window = smoothdata(optimal_threshold_per_window, 'movmean', 3);
        end
        
        % Clamp edges to first/last (smoothed) window values — fully local
        padded_centers    = [1, window_centers, N_epochs];
        padded_thresholds = [optimal_threshold_per_window(1), optimal_threshold_per_window, optimal_threshold_per_window(end)];
        
        % Ensure unique knot points
        [unique_centers, unique_idx] = unique(padded_centers);
        unique_thresholds = padded_thresholds(unique_idx);
        
        artifact_threshold_array = interp1(unique_centers, unique_thresholds, 1:N_epochs, 'makima');
    else
        artifact_threshold_array = repmat(optimal_threshold_per_window, 1, N_epochs);
    end
    
    % Ensure bounds
    artifact_threshold_array = max(minThreshold, min(maxThreshold, artifact_threshold_array));
    artifact_threshold = artifact_threshold_array;
    % Pre-calculate cosine weights for efficiency
    cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);
    
    artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
    if isempty(artifact_threshold_2)
        artifact_threshold_2 = artifact_threshold; % Fallback for 1-epoch edge case
    end
    
    [cleaned_data_1, artifacts_data_1, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec, cosine_weights, signal_type, 1, N_epochs, minThreshold);
    [cleaned_data_2, artifacts_data_2, ~] = clean_EEG(EEGdata_epoched_2, srate, epoch_size, artifact_threshold_2, refCOV, Eval_2, Evec_2, cosine_weights, signal_type, 1, N_epochs - 1, minThreshold);
    
    % Clear Stream 2 inputs as they are no longer needed
    clear EEGdata_epoched_2 Evec_2 Eval_2 COV_2;
    
    %% Combine the two processed streams using cosine weighting
    % cosine_weights is already calculated
    
    size_reconstructed_2 = size(cleaned_data_2, 2);
    if size_reconstructed_2 > 0
        sample_end = size_reconstructed_2 - shifting;
        % Apply weights to the second (shifted) stream
        cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
        cleaned_data_2(:, sample_end+1:end) = cleaned_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
        artifacts_data_2(:, 1:shifting) = artifacts_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
        artifacts_data_2(:, sample_end+1:end) = artifacts_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
    end
    
    % Combine streams (Optimize memory by clearing variables)
    cleaned_data = cleaned_data_1;
    clear cleaned_data_1; % Release memory
    
    artifacts_data = artifacts_data_1;
    clear artifacts_data_1; % Release memory
    
    if size_reconstructed_2 > 0
        cleaned_data(:, shifting+1:shifting+size_reconstructed_2) = cleaned_data(:, shifting+1:shifting+size_reconstructed_2) + cleaned_data_2;
    end
    clear cleaned_data_2; % Release memory
    
    if size_reconstructed_2 > 0
        artifacts_data(:, shifting+1:shifting+size_reconstructed_2) = artifacts_data(:, shifting+1:shifting+size_reconstructed_2) + artifacts_data_2;
    end
    clear artifacts_data_2; % Release memory
    
    % Remove padding to restore original data length
    cleaned_data = cleaned_data(:, 1:pnts_original);
    artifacts_data = artifacts_data(:, 1:pnts_original);
    
    %% Calculate final SENSAI score
    [~, ~, SENSAI_score] = SENSAI(mean(artifact_threshold_out), refCOV, Eval, Evec, noise_multiplier, COV, evecs_Template_cov, signal_type, SSI_top_PCs);
    
    % Calculate mean ENOVA for this band (average of per-epoch variance ratios)
    original_data = cleaned_data + artifacts_data;
    
    % Reshape into epochs (channels x samples x epochs)
    epoch_samples = round(srate * epoch_size);
    % Handle potential padding/truncation: use floor to get full epochs
    num_epochs_possible = floor(size(original_data, 2) / epoch_samples);
    len_to_use = num_epochs_possible * epoch_samples;
    
    original_epoched = reshape(original_data(:, 1:len_to_use), size(original_data, 1), epoch_samples, []);
    artifacts_epoched = reshape(artifacts_data(:, 1:len_to_use), size(artifacts_data, 1), epoch_samples, []);
    
    num_epochs_enova = size(original_epoched, 3);
    if num_epochs_enova > 0
        original_flat = reshape(original_epoched, [], num_epochs_enova);
        artifacts_flat = reshape(artifacts_epoched, [], num_epochs_enova);
        var_orig = var(original_flat, 0, 1);
        var_art = var(artifacts_flat, 0, 1);
        enova_per_epoch = zeros(1, num_epochs_enova);
        valid = var_orig > 0;
        enova_per_epoch(valid) = var_art(valid) ./ var_orig(valid);
        ENOVA = mean(enova_per_epoch);
    else
        ENOVA = 0;
    end
end
end

function padding = local_reflect_pad(eeg_data, samples_to_pad)
if samples_to_pad <= 0
    padding = zeros(size(eeg_data, 1), 0, 'like', eeg_data);
    return;
end

if size(eeg_data, 2) == 0
    error('Cannot pad empty EEG data.');
end

padding = zeros(size(eeg_data, 1), samples_to_pad, 'like', eeg_data);
filled = 0;
while filled < samples_to_pad
    reflected_chunk = fliplr(eeg_data);
    chunk_len = min(size(reflected_chunk, 2), samples_to_pad - filled);
    padding(:, filled + 1:filled + chunk_len) = reflected_chunk(:, 1:chunk_len);
    filled = filled + chunk_len;
end
end

function chunk_size = local_chunk_size(num_channels, epoch_samples, epoch_size, eeg_data)
bytes_per_scalar = 8;
if isa(eeg_data, 'single')
    bytes_per_scalar = 4;
end

time_limited_chunk_size = max(1, min(500, round(1000 / epoch_size)));

% Conservative peak-memory estimate per epoch for the sliding-window path:
% Evec + Eval + epoched input chunk + cleaned/artifact outputs + covariance summaries.
estimated_scalars_per_epoch = (2 * num_channels * num_channels) + (4 * num_channels * epoch_samples) + (2 * num_channels * num_channels);
target_chunk_bytes = 256 * 1024 * 1024;
memory_limited_chunk_size = max(1, floor(target_chunk_bytes / max(estimated_scalars_per_epoch * bytes_per_scalar, 1)));

chunk_size = max(1, min(time_limited_chunk_size, memory_limited_chunk_size));
end

function m_eff = local_compute_z3_noise_multiplier(Eval_sub, Evec_sub, refCOV_reg, evecs_Template_cov, SSI_top_PCs, minThreshold, epoch_samples)
% Closed-form zero-parameter Z3_SENSAI_Pure noise multiplier law:
%   Broadband (minThreshold == -4):
%     excess_iqr = max(1, iqr_log / (M - 1))
%     m_eff = clamp(((N_chans / M^2) * PR) / (iqr_log^2 * c_1 * excess_iqr^2),
%                   [max(1/M^2, (M - 1)/excess_iqr^3), 1.5 * M])
%   Wavelet bands (minThreshold ~= -4):
%     iqr_ref = M - clamp((c1_floor - c_1) / (1 - c1_floor), [0, 1])
%     excess_iqr = 1 + max(0, iqr_log - iqr_ref) / (M - 1)
%     m_eff = clamp((M * PR) / (iqr_log^2 * max(c_1, c1_floor) * excess_iqr^2),
%                   [max(1/M^2, M/excess_iqr^3), 1.5 * M])
% Computed over the upper-half active epochs (peak GEVD eigenvalue >= median)
% so partial temporal contamination is balanced against clean background epochs.

[N_chans, ~, w_len] = size(Eval_sub);
M = max(1, SSI_top_PCs);
if N_chans <= 2 || w_len < 1
    m_eff = double(M);
    return;
end
if nargin < 7 || isempty(epoch_samples)
    epoch_samples = (M^3) * N_chans;
end

all_evals_mat = zeros(N_chans, w_len);
part_ratios   = zeros(1, w_len);
top1_cos      = zeros(1, w_len);
Template_guess = double(evecs_Template_cov);
refCOV_reg_d   = double(refCOV_reg);

for epo = 1:w_len
    ev_raw = abs(double(diag(Eval_sub(:, :, epo))));
    [ev, ord] = sort(ev_raw, 'ascend');
    all_evals_mat(:, epo) = ev;
    ev_act = ev(2:end);
    part_ratios(epo) = (sum(ev_act)^2) / max(sum(ev_act.^2), 1e-300);
    
    v_top1 = double(Evec_sub(:, ord(end), epo));
    p_top1 = refCOV_reg_d * v_top1;
    top1_cos(epo) = norm(Template_guess' * (p_top1 / max(norm(p_top1), 1e-300)));
end

peak_ev = all_evals_mat(end, :);
topq_mask = (peak_ev >= median(peak_ev));
if ~any(topq_mask)
    topq_mask = true(1, w_len);
end

M_d = double(M);
m_min = M_d / max(M_d - 1.0, 1.0);
r_sub     = (M_d^3) / max(double(N_chans - 1), 1.0);
w_ch_dof  = 1.0 ./ (1.0 + exp(-(r_sub - 0.85) / 0.08));
w_ep_dof  = 1.0 ./ (1.0 + exp((epoch_samples / max(double(N_chans), 1.0) - (M_d + 1.25)) / 0.25));
w_dof     = w_ch_dof * w_ep_dof; n_dof = max(1, round(w_ch_dof * min(round(M_d)^3, floor(epoch_samples / (M_d + 1.0))) + (1.0 - w_ch_dof) * (round(M_d)^3)));
k_start = max(2, N_chans - n_dof + 1);
active_evals = all_evals_mat(k_start:end, topq_mask);
ev_floor = 1e-6 * median(peak_ev(topq_mask));
pos_evals = log(active_evals(active_evals > ev_floor));
if numel(pos_evals) >= 2
    iqr_log = max(0.5, prctile(pos_evals, 75) - prctile(pos_evals, 25));
else
    iqr_log = 1.0;
end

PR = mean(part_ratios(topq_mask));
c1 = max(0.15, min(0.95, mean(top1_cos(topq_mask))));

if minThreshold == -4
    scale_factor = double(N_chans) / (M_d^2);
    excess_iqr = max(1.0, iqr_log / (M_d - 1.0));
    m_raw = (scale_factor * PR) / ((iqr_log^2) * c1 * (excess_iqr^2));
    m_floor = max(m_min, (M_d - 1.0) / (excess_iqr^3));
    m_eff = max(m_floor, min(1.5 * M_d, m_raw));
else
    c1_floor = (M_d - 1.0) / M_d;
    w_nondip = min(1.0, max(0.0, (c1_floor - c1) / (1.0 - c1_floor)) * ((min(1.0, M_d / max(PR, 1e-12))) ^ w_dof));
    iqr_ref = M_d - w_nondip;
    excess_iqr = 1.0 + max(0.0, iqr_log - iqr_ref) / max(M_d - 1.0, 1.0);
    iso_boost = 1.0 + w_dof * max(0.0, (PR / M_d)^2 - 1.0);
    m_raw = (M_d * PR * iso_boost) / ((iqr_log^2) * max(c1, c1_floor) * (excess_iqr^2));
    m_floor = max(m_min, (M_d - w_dof * (M_d - m_min) * w_nondip) / (excess_iqr^3));
    m_ceil = (1.5 * M_d) * ((M_d / 1.5) ^ w_dof);
    m_eff = max(m_floor, min(m_ceil, m_raw));
end
end

function sc = local_strength_to_sc(S_val)
if isnan(S_val)
    sc = 0.5;
    return;
end
log_sc_knots = [-1.0, -log10(3), -log10(2), 0.0, 1.0];
S_knots      = [10.0,       8.0,       7.0, 6.0, 0.1];
sc = 10.0 .^ interp1(S_knots, log_sc_knots, double(S_val), 'linear', 'extrap');
end