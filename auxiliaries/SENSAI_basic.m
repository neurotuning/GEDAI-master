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

function [SENSAI_score, SIGNAL_subspace_similarity, NOISE_subspace_similarity, mean_ENOVA, ENOVA_per_epoch, signal_silhouette, SENSAI_normalized] = SENSAI_basic(signal_data, noise_data, srate, epoch_size, refCOV, NOISE_multiplier, signal_type)

    %   Calculates the Signal & Noise Subspace Alignment Index (SENSAI) from raw EEG data
    
regularization_lambda = 0.05;
reg_val = trace(refCOV) / length(refCOV);
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(length(refCOV), 'like', refCOV);

%% Estimate Signal Quality
num_chans = size(refCOV, 1);
epoch_samples = srate * epoch_size;

% Determine the number of top PCs to use based on signal type
if strcmpi(signal_type, 'meg')
    SSI_top_PCs = 4;
else
    SSI_top_PCs = 3;
end

% Compute the reference subspace from the top SSI_top_PCs eigenvectors of refCOV
[Vref, Dref] = eig(refCOV_reg);
[~, idxRef] = sort(diag(Dref), 'descend');
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
    [Vsig, Dsig] = eig(cov_signal_EEG);
    [~, idxSig] = sort(diag(Dsig), 'descend');
    basis_sig = Vsig(:, idxSig(1:SSI_top_PCs));
    cos_theta_sig = subspace_angles(basis_sig, basis_ref);
    % SSI = product of cosine similarities (= 1 when perfectly aligned)
    SIGNAL_subspace_similarity_distribution(epoch) = prod(cos_theta_sig);

    % NOISE SUBSPACE: top eigenvectors of noise covariance
    cov_noise = cov(noise_EEG_epoched(:,:,epoch)');
    [Vnoise, Dnoise] = eig(cov_noise);
    [~, idxNoise] = sort(diag(Dnoise), 'descend');
    basis_noise = Vnoise(:, idxNoise(1:SSI_top_PCs));
    cos_theta_noise = subspace_angles(basis_noise, basis_ref);
    NOISE_subspace_similarity_distribution(epoch) = prod(cos_theta_noise);

    % Explained Noise Variance (ENOVA)
    original_epoch = signal_EEG_epoched(:,:,epoch) + noise_EEG_epoched(:,:,epoch);
    var_original = var(original_epoch(:));
    var_noise = var(reshape(noise_EEG_epoched(:,:,epoch), [], 1));
    ENOVA_per_epoch(epoch) = var_noise / var_original;
end

mean_ENOVA = mean(ENOVA_per_epoch);
SIGNAL_subspace_similarity = 100 * mean(SIGNAL_subspace_similarity_distribution);
NOISE_subspace_similarity = 100 * mean(NOISE_subspace_similarity_distribution);

% --- Point-to-Centroid Normalized SENSAI ---
% Middle ground: accounts for individual point spreads relative to cluster centers.
% Bounded [-1, 1], but usually > 0 for successful denoising.
mu_sig = mean(SIGNAL_subspace_similarity_distribution);
mu_noise = mean(NOISE_subspace_similarity_distribution);
epsilon = 1e-6; % Avoid division by zero

% Distances of signal points to both centroids
a_sig = abs(SIGNAL_subspace_similarity_distribution - mu_sig);
b_sig = abs(SIGNAL_subspace_similarity_distribution - mu_noise);
% Silhouette per signal point
s_sig = (b_sig - a_sig) ./ max(max(a_sig, b_sig), epsilon);

% Distances of noise points to both centroids
a_noise = abs(NOISE_subspace_similarity_distribution - mu_noise);
b_noise = abs(NOISE_subspace_similarity_distribution - mu_sig);
% Silhouette per noise point
s_noise = (b_noise - a_noise) ./ max(max(a_noise, b_noise), epsilon);

% Final Normalized Score: Mean silhouette of the Signal cluster
% (Scaled to 0-1 range for consistency, where 0.5 is neutral)
SENSAI_normalized = 0.5 * (mean(s_sig) + 1);

% --- Signal Silhouette Score ---
% Only sensitive to the SSI separation (Y-axis in the visualization)
X_sil = [SIGNAL_subspace_similarity_distribution'; NOISE_subspace_similarity_distribution'];
Y_sil = [ones(numel(SIGNAL_subspace_similarity_distribution), 1); zeros(numel(NOISE_subspace_similarity_distribution), 1)];
try
    sil_scores = silhouette(X_sil, Y_sil, 'sqEuclidean');
    signal_silhouette = mean(sil_scores(Y_sil == 1));
catch
    signal_silhouette = NaN;
end

SENSAI_score = SIGNAL_subspace_similarity - NOISE_multiplier * NOISE_subspace_similarity;
end