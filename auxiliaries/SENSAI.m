function [SIGNAL_subspace_similarity, NOISE_subspace_similarity, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Eval, Evec, noise_multiplier, cov_total, refCOV_triu, signal_type)
% SENSAI Evaluates GEDAI cleaning quality using the Boundary-Aware Silhouette Score.

[cov_signal_epoched, cov_noise_epoched] = clean_SENSAI(artifact_threshold, refCOV, Eval, Evec, cov_total, signal_type);

%% 1. Calculate Principal Angles against Reference Subspace
num_chans = size(refCOV, 1);
num_epochs = size(cov_signal_epoched, 3);

if strcmpi(signal_type, 'meg')
    SSI_top_PCs = 4;
else
    SSI_top_PCs = 3;
end

[Vref, Dref] = eig(refCOV);
[~, idxRef] = sort(diag(Dref), 'descend');
basis_ref = Vref(:, idxRef(1:SSI_top_PCs));

clean_angles = zeros(num_epochs, SSI_top_PCs);
artifact_angles = zeros(num_epochs, SSI_top_PCs);

for epoch = 1:num_epochs
    % SIGNAL SUBSPACE
    cov_signal = cov_signal_epoched(:,:,epoch);
    [Vsig, Dsig] = eig(cov_signal);
    [~, idxSig] = sort(diag(Dsig), 'descend');
    basis_sig = Vsig(:, idxSig(1:SSI_top_PCs));
    cos_theta_sig = subspace_angles(basis_sig, basis_ref);
    clean_angles(epoch, :) = cos_theta_sig(:)';
    
    % NOISE SUBSPACE
    cov_noise = cov_noise_epoched(:,:,epoch);
    if all(cov_noise(:) == 0)
        artifact_angles(epoch, :) = 0;
    else
        [Vnoise, Dnoise] = eig(cov_noise);
        [~, idxNoise] = sort(diag(Dnoise), 'descend');
        basis_noise = Vnoise(:, idxNoise(1:SSI_top_PCs));
        cos_theta_noise = subspace_angles(basis_noise, basis_ref);
        artifact_angles(epoch, :) = cos_theta_noise(:)';
    end
end

%% 2. Compute SSI Components (For auxiliary logging)
SIGNAL_subspace_similarity = mean(prod(clean_angles, 2));
NOISE_subspace_similarity  = mean(prod(artifact_angles, 2));

%% 3. Compute Boundary-Aware Silhouette Score
% This score measures how well the Signal cluster is isolated from the Noise cluster.
X_sig = clean_angles;
X_noise = artifact_angles;

% a(i): Internal Cohesion (Mean Euclidean distance to all other Signal points)
% Vectorized: Dist^2 = sum(A^2) + sum(B^2) - 2AB
D_sig = sqrt(max(0, bsxfun(@plus, sum(X_sig.^2, 2), sum(X_sig.^2, 2)') - 2*(X_sig*X_sig')));
if num_epochs > 1
    a_i = (sum(D_sig, 2)) ./ (num_epochs - 1);
else
    a_i = zeros(num_epochs, 1);
end

% b(i): Boundary Separation (Distance to the single closest Noise point)
if any(X_noise(:))
    D_bg = sqrt(max(0, bsxfun(@plus, sum(X_sig.^2, 2), sum(X_noise.^2, 2)') - 2*(X_sig*X_noise')));
    b_i = min(D_bg, [], 2);
else
    % If no artifacts were removed, distance is to the origin [0,0,0]
    b_i = sqrt(sum(X_sig.^2, 2)); 
end

% s(i) = (b - a) / max(a, b)
s_i = (b_i - a_i) ./ max(a_i, b_i);
s_i(isnan(s_i)) = 0; 

% Final SENSAI_score: Multiplicative Accuracy & Isolation
% 1. Clamp s_i to positive values (0.01 floor for stability)
s_i_clamped = max(0.01, s_i);

% 2. Use noise_multiplier as an "Isolation Strictness" factor.
% High multiplier = High requirement for a 'clear zone' around the signal.
% Centered at 3 (Balanced), so exponent is multiplier/3.
isolation_strictness = s_i_clamped .^ (noise_multiplier / 3);

% 3. Combine with Signal Similarity
% Score = Mean of [Accuracy * StrictIsolation]
SENSAI_score = mean( isolation_strictness .* prod(clean_angles, 2) );

end