<<<<<<< Updated upstream
function [SIGNAL_subspace_similarity, NOISE_subspace_similarity, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Eval, Evec, noise_multiplier, cov_total, refCOV_triu, signal_type)
=======
function [SIGNAL_subspace_similarity, NOISE_subspace_similarity, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Eval, Evec, noise_multiplier, cov_total, evecs_Template_cov, signal_type, SSI_top_PCs)
>>>>>>> Stashed changes
% SENSAI Evaluates GEDAI cleaning quality using the Boundary-Aware Silhouette Score.

[cov_signal_epoched, cov_noise_epoched] = clean_SENSAI(artifact_threshold, refCOV, Eval, Evec, cov_total, signal_type);

<<<<<<< Updated upstream
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
=======
num_chans = size(refCOV, 1);
num_epochs = size(cov_signal_epoched, 3);

% Use eig (full) for small matrices, eigs for large
use_full_eig = (num_chans < 150);

% Storage for SSI vectors (needed for Silhouette)
X_sig = zeros(num_epochs, SSI_top_PCs);
X_noise = zeros(num_epochs, SSI_top_PCs);

for epoch = 1:num_epochs
    % SIGNAL SUBSPACE
    cov_signal = cov_signal_epoched(:,:,epoch);
    if use_full_eig
        [Vs, Ds] = eig(cov_signal);
        [~, idx] = sort(diag(Ds), 'descend');
        evecs_s = Vs(:, idx(1:SSI_top_PCs));
    else
        [evecs_s, ~] = eigs(cov_signal, SSI_top_PCs);
    end
    X_sig(epoch, :) = subspace_angles(evecs_s, evecs_Template_cov)';
    
    % NOISE SUBSPACE
    cov_noise = cov_noise_epoched(:,:,epoch);
    if all(cov_noise(:) == 0)
        X_noise(epoch, :) = 0;
    else
        if use_full_eig
            [Vn, Dn] = eig(cov_noise);
            [~, idx] = sort(diag(Dn), 'descend');
            evecs_n = Vn(:, idx(1:SSI_top_PCs));
        else
            [evecs_n, ~] = eigs(cov_noise, SSI_top_PCs);
        end
        X_noise(epoch, :) = subspace_angles(evecs_n, evecs_Template_cov)';
    end
end

%% 1. Compute Classical SSI Metrics
ssi_sig = prod(X_sig, 2);
ssi_noise = prod(X_noise, 2);

mean_ssi_sig = mean(ssi_sig);
mean_ssi_noise = mean(ssi_noise);

%% 2. Calculate Power Fracts (For visualization parity)
pow_sig = zeros(num_epochs, 1);
pow_noise = zeros(num_epochs, 1);
pow_tot = zeros(num_epochs, 1);
for e = 1:num_epochs
    pow_sig(e) = trace(cov_signal_epoched(:, :, e));
    pow_noise(e) = trace(cov_noise_epoched(:, :, e));
    pow_tot(e) = trace(cov_total(:, :, e));
end
pow_tot(pow_tot == eps) = 1; % guard

pow_fract_sig = pow_sig ./ pow_tot;
pow_fract_noise = pow_noise ./ pow_tot;

%% 3. Calculate 2D Fisher Separation J
% This mimics the exact LDA mechanism seen in the plots
valid_noise = pow_noise > eps;
if sum(valid_noise) > 1
    X_sig_2d = [ssi_sig, pow_fract_sig];
    X_noise_2d = [ssi_noise(valid_noise), pow_fract_noise(valid_noise)];
    
    m_sig = mean(X_sig_2d, 1);
    m_noise = mean(X_noise_2d, 1);
    
    S_w = cov(X_sig_2d) + cov(X_noise_2d) + eye(2)*1e-4; % Regularized within-class
    diff_m = m_sig - m_noise;
    S_b = diff_m' * diff_m; % Between-class
    
    Fisher_J = trace(S_w \ S_b);
else
    Fisher_J = 0; % No separation if no noise cluster
end

%% 4. Apply Fisher-Discounted Topological Penalty
% Map Fisher Separation (0 to Inf) to a discount factor [0, 1]
% J=0 -> Discount=0 (Full Penalty). J=20 -> Discount ~0.86 (Relief)
separation_factor = 1 - exp(-0.1 * Fisher_J);

% The classical strictness multiplier is relaxed IF the 2D cluster is physically isolated!
effective_penalty = mean_ssi_noise * (1 - separation_factor);

SIGNAL_subspace_similarity = mean_ssi_sig * 100;
NOISE_subspace_similarity = effective_penalty * 100;
SENSAI_score = SIGNAL_subspace_similarity - (noise_multiplier * NOISE_subspace_similarity);
>>>>>>> Stashed changes

end