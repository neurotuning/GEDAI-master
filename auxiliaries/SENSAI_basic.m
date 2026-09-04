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

function [SENSAI_score, SIGNAL_subspace_similarity, NOISE_subspace_similarity, mean_ENOVA, ENOVA_per_epoch] = SENSAI_basic(signal_data, noise_data, srate, epoch_size, refCOV, NOISE_multiplier, signal_type)

    %   Calculates the Signal & Noise Subspace Alignment Index (SENSAI) from raw EEG data
    
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;
regularization_lambda = 0.05;
reg_val = trace(refCOV) / length(refCOV);
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(length(refCOV), 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;

%% Estimate Signal Quality
num_chans = size(refCOV, 1);
epoch_samples = round(srate * epoch_size);

if strcmpi(signal_type, 'meg')
    % Option 12: Adaptive GEVD Artifact Spectrum Pre-Scan
    cov_data_prescan = cov(double(signal_data + noise_data)');
    cov_data_prescan = (cov_data_prescan + cov_data_prescan') / 2;
    [~, D_gevd_prescan] = eig(cov_data_prescan, refCOV_reg);
    evals_prescan = sort(diag(D_gevd_prescan), 'descend');
    med_val_prescan = median(evals_prescan);
    if med_val_prescan > 0
        norm_evals = evals_prescan / med_val_prescan;
    else
        norm_evals = evals_prescan;
    end
    if length(norm_evals) >= 3 && (norm_evals(3) > 200 || (norm_evals(1) > 0 && norm_evals(2) / norm_evals(1) > 0.20 && norm_evals(1) < 12000))
        SSI_top_PCs = min(3, num_chans);
    else
        SSI_top_PCs = min(2, num_chans);
    end
else
    SSI_top_PCs = min(3, num_chans);
end
[Vref, Dref] = eig(refCOV_reg);
[~, idxRef] = sort(diag(Dref), 'descend');
SSI_top_PCs = min(SSI_top_PCs, size(Vref, 2));
basis_ref = Vref(:, idxRef(1:SSI_top_PCs));

% --- FIX START: Truncate data to contain a whole number of epochs ---
pnts = size(signal_data, 2);
num_epochs_possible = floor(pnts / epoch_samples);
new_length = num_epochs_possible * epoch_samples;

signal_data = signal_data(:, 1:new_length);
noise_data = noise_data(:, 1:new_length);
% --- FIX END ---

% Epoch signal and noise data
signal_EEG_epoched = reshape(signal_data, num_chans, epoch_samples, []);
noise_EEG_epoched = reshape(noise_data, num_chans, epoch_samples, []);
num_epochs = size(signal_EEG_epoched, 3);
SIGNAL_subspace_similarity_distribution = zeros(1, num_epochs);
NOISE_subspace_similarity_distribution = zeros(1, num_epochs);
ENOVA_per_epoch = zeros(1, num_epochs);

for epoch = 1:num_epochs
    % SIGNAL SUBSPACE: top eigenvectors of signal covariance
    cov_signal_EEG = cov(signal_EEG_epoched(:,:,epoch)');
    cov_signal_EEG = (cov_signal_EEG + cov_signal_EEG') / 2;
    [Vsig, Dsig] = eig(cov_signal_EEG);
    [~, idxSig] = sort(diag(Dsig), 'descend');
    basis_sig = Vsig(:, idxSig(1:SSI_top_PCs));
    cos_theta_sig = subspace_angles(basis_sig, basis_ref);
    % SSI = product of cosine similarities (= 1 when perfectly aligned)
    SIGNAL_subspace_similarity_distribution(epoch) = prod(cos_theta_sig);

    % NOISE SUBSPACE
    cov_noise = cov(noise_EEG_epoched(:,:,epoch)');
    cov_noise = (cov_noise + cov_noise') / 2;
    [Vnoise, Dnoise] = eig(cov_noise);
    [~, idxNoise] = sort(diag(Dnoise), 'descend');
    basis_noise = Vnoise(:, idxNoise(1:SSI_top_PCs));
    if strcmpi(signal_type, 'meg')
        s_noise = svd(basis_noise' * basis_ref);
        NOISE_subspace_similarity_distribution(epoch) = sum(s_noise.^6);
    else
        cos_theta_noise = subspace_angles(basis_noise, basis_ref);
        NOISE_subspace_similarity_distribution(epoch) = prod(cos_theta_noise);
    end

    % Explained Noise Variance (ENOVA)
    original_epoch = signal_EEG_epoched(:,:,epoch) + noise_EEG_epoched(:,:,epoch);
    var_original = var(original_epoch(:));
    var_noise = var(reshape(noise_EEG_epoched(:,:,epoch), [], 1));
    ENOVA_per_epoch(epoch) = var_noise / var_original;
end

mean_ENOVA = mean(ENOVA_per_epoch);
SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_similarity_distribution);
NOISE_subspace_similarity = 100 * mean(NOISE_subspace_similarity_distribution);
SENSAI_score = SIGNAL_subspace_similarity - NOISE_multiplier * NOISE_subspace_similarity;
end