function [B_final, B_sub, A_clean, clean_filters] = GEDAI_create_subject_refCOV(EEG, clean_filters, G_norm, alpha, regularization_lambda, num_comp_per_epoch, srate, epoch_size, blending_method)
% GEDAI_CREATE_SUBJECT_REFCOV Constructs a subject-specific empirical reference 
% covariance matrix from cleanest epoch forward patterns regularized with a leadfield prior.
%
% Usage:
%   [B_final, B_sub, A_clean, clean_filters] = GEDAI_create_subject_refCOV(EEG, [], G_norm, alpha, regularization_lambda, num_comp_per_epoch, srate, epoch_size, blending_method)
%
% Inputs:
%   EEG                   - EEGLAB struct (continuous or epoched) or [N_chan x N_samples x K_epochs] data matrix.
%   clean_filters         - (Optional) Matrix of spatial filters per epoch. If empty, GEVD is run per epoch against G_norm.
%   G_norm                - [N_chan x N_chan] theoretical leadfield Gram matrix (refCOV).
%   alpha                 - (Optional) Blending parameter between empirical patterns and G_norm. Default: 0.5
%   regularization_lambda - (Optional) Diagonal isotropic shrinkage parameter. Default: 0.05
%   num_comp_per_epoch    - (Optional) Number of cleanest signal components to retain per epoch. Default: 3
%   srate                 - (Optional) Sampling rate in Hz. Default: EEG.srate or 250
%   epoch_size            - (Optional) Epoch size in seconds for continuous data windowing. Default: 1
%   blending_method       - (Optional) 'procrustes' (default), 'riemannian', or 'linear'.
%
% Outputs:
%   B_final               - [N_chan x N_chan] Regularized subject-specific reference covariance matrix.
%   B_sub                 - [N_chan x N_chan] Pure empirical reference covariance matrix (unregularized).
%   A_clean               - [N_chan x (K_epochs * num_comp_per_epoch)] Matrix of column-concatenated clean forward patterns.
%   clean_filters         - Spatial filters extracted per epoch.

if nargin < 3 || isempty(G_norm)
    error('G_norm (theoretical leadfield reference matrix) is required.');
end
if nargin < 4 || isempty(alpha)
    alpha = 0.5; % Default to 50% empirical, 50% leadfield (Riemannian geodesic midpoint)
end
if nargin < 5 || isempty(regularization_lambda)
    regularization_lambda = 0.05; % Standardized 5% isotropic shrinkage
end
if nargin < 6 || isempty(num_comp_per_epoch)
    num_comp_per_epoch = 3; % Default: top 3 signal components per epoch
end

% Extract data array and srate
if isstruct(EEG) && isfield(EEG, 'data')
    data = double(EEG.data);
    if (nargin < 7 || isempty(srate)) && isfield(EEG, 'srate') && ~isempty(EEG.srate)
        srate = EEG.srate;
    end
else
    data = double(EEG);
end

if nargin < 7 || isempty(srate)
    srate = 250;
end
if nargin < 8 || isempty(epoch_size)
    epoch_size = 1;
end
if nargin < 9 || isempty(blending_method)
    blending_method = 'gromov-wasserstein'; % Default to Gromov-Wasserstein optimal transport blending
end

[N_chan, N_pts, N_epochs] = size(data);

% Automatic windowing for continuous 2D data
if N_epochs == 1 && N_pts >= round(srate * epoch_size)
    epoch_samples = round(srate * epoch_size);
    num_epochs = floor(N_pts / epoch_samples);
    if num_epochs >= 1
        data = reshape(data(:, 1:num_epochs*epoch_samples), N_chan, epoch_samples, num_epochs);
        [N_chan, N_pts, N_epochs] = size(data);
    end
end

num_comp_per_epoch = min(num_comp_per_epoch, N_chan);

% Ensure G_norm is symmetric and apply standardized isotropic shrinkage for GEVD
G_norm = real(G_norm);
G_norm = (G_norm + G_norm') / 2;
reg_val_G = trace(G_norm) / N_chan;
G_reg = (1 - regularization_lambda) * G_norm + regularization_lambda * reg_val_G * eye(N_chan);
G_reg = (G_reg + G_reg') / 2;

% Precompute Cholesky factor of G_reg once for accelerated symmetric eigensolving
try
    R_chol = chol(G_reg);
    has_chol = true;
catch
    has_chol = false;
end

A_clean_list = cell(1, N_epochs);
clean_filters_cell = cell(1, N_epochs);

% Pass 1: Solve GEVD C_k * v = lambda * G_reg * v per epoch and extract top N signal components
for k = 1:N_epochs
    if N_epochs == 1
        X_k = data;
    else
        X_k = data(:, :, k);
    end
    X_k_zero = X_k - mean(X_k, 2);
    C_k = (X_k_zero * X_k_zero') / max(1, size(X_k, 2) - 1);
    C_k = (C_k + C_k') / 2;
    
    if nargin >= 2 && ~isempty(clean_filters)
        if size(clean_filters, 2) == N_epochs
            v_k_set = clean_filters(:, k);
        else
            v_k_set = clean_filters(:, (k-1)*num_comp_per_epoch + (1:num_comp_per_epoch));
        end
    else
        if has_chol
            % Accelerated symmetric eigenvalue decomposition via pre-factored Cholesky
            C_tilde = (R_chol') \ (C_k / R_chol);
            C_tilde = (C_tilde + C_tilde') / 2;
            [V_tilde, D_eig] = eig(C_tilde, 'vector');
            [~, sort_idx] = sort(D_eig, 'ascend');
            y_sel = V_tilde(:, sort_idx(1:num_comp_per_epoch));
            v_k_set = R_chol \ y_sel;
        else
            % Fallback to standard QZ generalized eigenvalue decomposition
            [V_eig, D_eig] = eig(C_k, G_reg);
            [~, sort_idx] = sort(diag(D_eig), 'ascend');
            v_k_set = V_eig(:, sort_idx(1:num_comp_per_epoch));
        end
    end
    
    clean_filters_cell{k} = v_k_set;
    
    % Vectorized computation of physical forward patterns a_m for selected components
    Cv = C_k * v_k_set;
    denom = sum(v_k_set .* Cv, 1);
    denom(abs(denom) < 1e-12) = 1;
    a_k_set = Cv ./ denom;
    
    A_clean_list{k} = a_k_set;
end

% Column concatenation across all epochs
A_clean = cell2mat(A_clean_list);
clean_filters = cell2mat(clean_filters_cell);

% Aggregate Outer Product B_sub = (1 / Total_Patterns) * A_clean * A_clean'
total_patterns = size(A_clean, 2);
B_sub = (A_clean * A_clean') / total_patterns;
B_sub = (B_sub + B_sub') / 2;

% Scale Matching: normalize trace of G_norm to match B_sub so scale differences don't distort blending
tr_B = trace(B_sub);
tr_G = trace(G_norm);
if tr_G > 0 && tr_B > 0
    G_scaled = G_norm * (tr_B / tr_G);
else
    G_scaled = G_norm;
end

if alpha == 0
    % Pure empirical, apply standard shrinkage
    reg_val_B = tr_B / N_chan;
    B_final = (1 - regularization_lambda) * B_sub + regularization_lambda * reg_val_B * eye(N_chan);

elseif alpha == 1
    % Pure leadfield prior, apply standard shrinkage
    reg_val_G_sc = trace(G_scaled) / N_chan;
    B_final = (1 - regularization_lambda) * G_scaled + regularization_lambda * reg_val_G_sc * eye(N_chan);

elseif strcmpi(blending_method, 'procrustes')
    % --- Eigenvalue-Weighted Orthogonal Procrustes Alignment (SO(N)) & Eigenvalue Blending ---
    [U_B, D_B] = eig(B_sub);
    [U_G, D_G] = eig(G_scaled);
    d_B = diag(D_B);
    d_G = diag(D_G);
    [d_B, idx_B] = sort(d_B, 'descend');
    U_B = U_B(:, idx_B);
    [d_G, idx_G] = sort(d_G, 'descend');
    U_G = U_G(:, idx_G);

    % Factor / Energy weighting: scale eigenvectors by square root of their eigenvalues
    % This aligns high-power physiological modes while preventing noise axes from distorting the rotation
    W_G = diag(sqrt(max(0, d_G)));
    W_B = diag(sqrt(max(0, d_B)));
    M_weighted = W_G * (U_G' * U_B) * W_B;

    % Find optimal rigid rotation R in SO(N) weighted by dominant signal modes
    [U_svd, ~, V_svd] = svd(M_weighted);
    if det(U_svd * V_svd') < 0
        V_svd(:, end) = -V_svd(:, end);
    end
    R = U_svd * V_svd';
    U_aligned = U_G * R; % Aligned eigenspace in subject coordinate frame

    % Blend eigenvalues along alpha
    d_blend = (1 - alpha) * d_B + alpha * d_G;

    % Reconstruct aligned reference matrix
    B_proc = U_aligned * diag(d_blend) * U_aligned';
    B_proc = (B_proc + B_proc') / 2;

    % Scale matching and standardized isotropic shrinkage
    tr_proc = trace(B_proc);
    if tr_proc > 0 && tr_B > 0
        B_proc = B_proc * (tr_B / tr_proc);
    end
    reg_val = trace(B_proc) / N_chan;
    B_final = (1 - regularization_lambda) * B_proc + regularization_lambda * reg_val * eye(N_chan);

elseif strcmpi(blending_method, 'gromov-wasserstein')
    % --- Entropic Fused Gromov-Wasserstein Optimal Transport Alignment ---
    % Finds a soft coupling matrix T that preserves the internal geometric
    % structure (pairwise correlation distances) of both B_sub and G_scaled,
    % then uses T as a linear alignment operator.

    % Correlation distance matrices (scale-invariant internal structure)
    % Source = leadfield (thing to warp), Target = empirical (frame to match)
    D_A = cov2dist_gedai(G_scaled);
    D_B = cov2dist_gedai(B_sub);

    % Uniform marginal distributions over sensors
    p_marg = ones(N_chan, 1) / N_chan;
    q_marg = ones(N_chan, 1) / N_chan;

    % No prior cross-domain cost (alignment is purely structure-driven)
    M_cross = zeros(N_chan, N_chan);

    % Fixed entropic regularization (moderate smoothing, avoids nested BayesOpt)
    eps_reg = 0.01;

    % Solve for optimal transport coupling
    T = entropic_fgw_sinkhorn(D_A, D_B, M_cross, p_marg, q_marg, eps_reg, alpha, 120);

    % Rescale coupling to linear alignment operator and warp leadfield into subject space
    W_align = N_chan * T;
    B_ot = W_align * G_scaled * W_align';
    B_ot = (B_ot + B_ot') / 2;

    % Scale matching to empirical trace and isotropic shrinkage
    tr_ot = trace(B_ot);
    if tr_ot > 0 && tr_B > 0
        B_ot = B_ot * (tr_B / tr_ot);
    end
    reg_val = trace(B_ot) / N_chan;
    B_final = (1 - regularization_lambda) * B_ot + regularization_lambda * reg_val * eye(N_chan);

elseif strcmpi(blending_method, 'linear')
    % Convex linear blending between regularized B_sub and G_scaled
    reg_val_B = tr_B / N_chan;
    B_sub_spd = (1 - regularization_lambda) * B_sub + regularization_lambda * reg_val_B * eye(N_chan);
    reg_val_G = trace(G_scaled) / N_chan;
    G_spd = (1 - regularization_lambda) * G_scaled + regularization_lambda * reg_val_G * eye(N_chan);
    B_final = (1 - alpha) * B_sub_spd + alpha * G_spd;

else
    % --- Scale-Invariant Riemannian Geodesic Interpolation on SPD Manifold ---
    reg_val_B = tr_B / N_chan;
    B_sub_spd = (1 - regularization_lambda) * B_sub + regularization_lambda * reg_val_B * eye(N_chan);
    B_sub_spd = (B_sub_spd + B_sub_spd') / 2;

    reg_val_G = trace(G_scaled) / N_chan;
    G_spd = (1 - regularization_lambda) * G_scaled + regularization_lambda * reg_val_G * eye(N_chan);
    G_spd = (G_spd + G_spd') / 2;

    [Vg, Dg] = eig(G_spd);
    dg = diag(Dg);
    eps_g = max(dg) * 1e-12;
    dg(dg < eps_g) = eps_g;
    G_sqrt = Vg * diag(sqrt(dg)) * Vg';
    G_inv_sqrt = Vg * diag(1 ./ sqrt(dg)) * Vg';

    M_mid = G_inv_sqrt * B_sub_spd * G_inv_sqrt;
    M_mid = (M_mid + M_mid') / 2;

    [Vm, Dm] = eig(M_mid);
    dm = diag(Dm);
    eps_m = max(dm) * 1e-12;
    dm(dm < eps_m) = eps_m;

    M_mid_pow = Vm * diag(dm .^ (1 - alpha)) * Vm';
    B_riem = G_sqrt * M_mid_pow * G_sqrt;

    tr_riem = trace(B_riem);
    if tr_riem > 0 && tr_B > 0
        B_riem = B_riem * (tr_B / tr_riem);
    end
    B_final = B_riem;
end

% Ensure matrix is real and symmetric
B_final = real(B_final);
B_final = (B_final + B_final') / 2;

end


%% --- Entropic Fused Gromov-Wasserstein via Projected Sinkhorn Iterations ---
function T = entropic_fgw_sinkhorn(D_A, D_B, M_cross, p, q, eps_reg, alpha, max_iter)
% Solves the Fused Gromov-Wasserstein problem using mirror descent with
% Sinkhorn projections. The quadratic GW term is linearized at each step,
% then entropic matrix scaling finds the next coupling iterate.
%
% Inputs:
%   D_A       - [n x n] intra-domain distance matrix for source (empirical)
%   D_B       - [m x m] intra-domain distance matrix for target (leadfield)
%   M_cross   - [n x m] cross-domain cost matrix (zeros if no prior)
%   p, q      - marginal distributions (column vectors summing to 1)
%   eps_reg   - entropic regularization strength
%   alpha     - trade-off: (1-alpha)*M_cross + alpha*Grad_GW
%   max_iter  - maximum outer Frank-Wolfe / mirror descent iterations
%
% Output:
%   T         - [n x m] optimal transport coupling matrix

    % Initialize coupling with uniform independent distribution
    T = p * q';

    % Precompute constant terms of the GW gradient
    f1_A = D_A.^2 * p;
    f2_B = (q' * (D_B.^2)')';

    tol = 1e-6;
    for iter = 1:max_iter
        T_old = T;

        % 1. Compute linearized GW gradient contracted with current T
        Grad_GW = bsxfun(@plus, f1_A, f2_B') - 2 * (D_A * T * (D_B'));

        % Combined Fused Cost
        C_total = (1 - alpha) * M_cross + alpha * Grad_GW;

        % 2. Inner Sinkhorn iteration (log-domain stabilized)
        K = exp(-C_total / eps_reg);
        K = max(K, realmin); % Numerical underflow protection

        u = ones(size(p));
        for s_iter = 1:60
            v = q ./ (K' * u + 1e-12);
            u = p ./ (K * v + 1e-12);
        end
        T_sinkhorn = diag(u) * K * diag(v);

        % Damped update for convergence stability in non-convex quadratic OT
        T = 0.6 * T + 0.4 * T_sinkhorn;

        if norm(T - T_old, 'fro') < tol
            break;
        end
    end
end


%% --- Helper: Trace-Normalized Covariance Distance ---
function D = cov2dist_gedai(C)
% Converts a covariance matrix into a trace-normalized pairwise Euclidean
% distance matrix over sensor topography columns. Trace normalization
% preserves the physical spatial decay of leadfield dipoles while making
% the global scale dimensionless (unit total power).
    C_norm = C / trace(C);
    n = size(C_norm, 1);
    col_norms_sq = sum(C_norm.^2, 1); % 1 x n
    D_sq = bsxfun(@plus, col_norms_sq', col_norms_sq) - 2 * (C_norm' * C_norm);
    D_sq = max(D_sq, 0); % Protect against roundoff
    D = sqrt(D_sq);
    D = D / max(D(:)); % Normalize for stable Sinkhorn scaling
end


