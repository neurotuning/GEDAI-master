function [B_final, B_sub, A_clean, clean_filters] = GEDAI_create_subject_refCOV(EEG, clean_filters, G_norm, alpha, gamma, num_comp_per_epoch, srate, epoch_size, blending_method)
% GEDAI_CREATE_SUBJECT_REFCOV Constructs a subject-specific empirical reference 
% covariance matrix from cleanest epoch forward patterns regularized with a leadfield prior.
%
% Usage:
%   [B_final, B_sub, A_clean, clean_filters] = GEDAI_create_subject_refCOV(EEG, [], G_norm, alpha, gamma, num_comp_per_epoch, srate, epoch_size, blending_method)
%
% Inputs:
%   EEG                  - EEGLAB struct (continuous or epoched) or [N_chan x N_samples x K_epochs] data matrix.
%   clean_filters        - (Optional) Matrix of spatial filters per epoch. If empty, GEVD is run per epoch against G_norm.
%   G_norm               - [N_chan x N_chan] theoretical leadfield Gram matrix (refCOV).
%   alpha                - (Optional) Blending parameter between empirical patterns and G_norm. Default: 0
%   gamma                - (Optional) Diagonal regularization floor. Default: 1e-6
%   num_comp_per_epoch   - (Optional) Number of cleanest signal components to retain per epoch. Default: 3
%   srate                - (Optional) Sampling rate in Hz. Default: EEG.srate or 250
%   epoch_size           - (Optional) Epoch size in seconds for continuous data windowing. Default: 1
%   blending_method      - (Optional) 'procrustes' (default), 'riemannian', or 'linear'.
%
% Outputs:
%   B_final              - [N_chan x N_chan] Regularized subject-specific reference covariance matrix.
%   B_sub                - [N_chan x N_chan] Pure empirical reference covariance matrix (unregularized).
%   A_clean              - [N_chan x (K_epochs * num_comp_per_epoch)] Matrix of column-concatenated clean forward patterns.
%   clean_filters        - Spatial filters extracted per epoch.

if nargin < 3 || isempty(G_norm)
    error('G_norm (theoretical leadfield reference matrix) is required.');
end
if nargin < 4 || isempty(alpha)
    alpha = 0.5; % Default to 50% empirical, 50% leadfield (Riemannian geodesic midpoint)
end
if nargin < 5 || isempty(gamma)
    gamma = 1e-6;
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
    blending_method = 'procrustes'; % Default to Procrustes-aligned eigenvalue blending
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

% Ensure G_norm is symmetric and non-singular for GEVD
G_norm = real(G_norm);
G_norm = (G_norm + G_norm') / 2;
G_reg = G_norm + 1e-6 * trace(G_norm) / N_chan * eye(N_chan);

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
        % Solve GEVD: C_k * v = lambda * G_reg * v
        [V_eig, D_eig] = eig(C_k, G_reg);
        % Smallest eigenvalues = cleanest neural components (highest SNR relative to leadfield prior)
        [~, sort_idx] = sort(diag(D_eig), 'ascend');
        v_k_set = V_eig(:, sort_idx(1:num_comp_per_epoch));
    end
    
    clean_filters_cell{k} = v_k_set;
    
    % Compute physical forward pattern a_m for each selected component
    a_k_set = zeros(N_chan, size(v_k_set, 2));
    for m = 1:size(v_k_set, 2)
        v_km = v_k_set(:, m);
        denom = v_km' * C_k * v_km;
        if abs(denom) < 1e-12
            a_k_set(:, m) = C_k * v_km;
        else
            a_k_set(:, m) = (C_k * v_km) / denom;
        end
    end
    
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
    % Pure empirical, no blending needed
    B_final = B_sub + gamma * eye(N_chan);

elseif alpha == 1
    % Pure leadfield prior
    B_final = G_scaled + gamma * eye(N_chan);

elseif strcmpi(blending_method, 'procrustes') && alpha > 0 && alpha < 1
    % --- Procrustes-Aligned Eigenvalue Blending ---
    % Eigendecompose both matrices
    B_sub_reg = B_sub + 1e-10 * tr_B / N_chan * eye(N_chan);
    G_scaled_reg = G_scaled + 1e-10 * tr_B / N_chan * eye(N_chan);
    B_sub_reg = (B_sub_reg + B_sub_reg') / 2;
    G_scaled_reg = (G_scaled_reg + G_scaled_reg') / 2;

    [U_B, D_B] = eig(B_sub_reg);
    [U_G, D_G] = eig(G_scaled_reg);
    d_B = diag(D_B);
    d_G = diag(D_G);

    % Sort both by descending eigenvalue for consistent axis ordering
    [d_B, idx_B] = sort(d_B, 'descend');
    U_B = U_B(:, idx_B);
    [d_G, idx_G] = sort(d_G, 'descend');
    U_G = U_G(:, idx_G);

    % Orthogonal Procrustes: find rotation R that best aligns U_B to U_G
    [U_svd, ~, V_svd] = svd(U_B' * U_G);
    R = U_svd * V_svd';

    % Rotate subject eigenvectors into leadfield-aligned frame
    U_aligned = U_B * R;

    % Blend eigenvalues in the aligned coordinate system
    d_B(d_B < 1e-12) = 1e-12;
    d_G(d_G < 1e-12) = 1e-12;
    d_blend = (1 - alpha) * d_B + alpha * d_G;

    % Reconstruct blended matrix
    B_proc = U_aligned * diag(d_blend) * U_aligned';

    % Re-scale trace to guarantee strict scale conservation
    tr_proc = trace(B_proc);
    if tr_proc > 0 && tr_B > 0
        B_proc = B_proc * (tr_B / tr_proc);
    end
    B_final = B_proc + gamma * eye(N_chan);

elseif strcmpi(blending_method, 'riemannian') && alpha > 0 && alpha < 1
    % --- Scale-Invariant Riemannian Geodesic Interpolation on SPD Manifold ---
    G_reg_blend = G_scaled + 1e-8 * tr_B / N_chan * eye(N_chan);
    G_reg_blend = (G_reg_blend + G_reg_blend') / 2;

    [Vg, Dg] = eig(G_reg_blend);
    dg = diag(Dg);
    dg(dg < 1e-12) = 1e-12;
    G_sqrt = Vg * diag(sqrt(dg)) * Vg';
    G_inv_sqrt = Vg * diag(1 ./ sqrt(dg)) * Vg';

    % Project B_sub onto G-tangent space
    M_mid = G_inv_sqrt * B_sub * G_inv_sqrt;
    M_mid = (M_mid + M_mid') / 2;

    [Vm, Dm] = eig(M_mid);
    dm = diag(Dm);
    dm(dm < 1e-12) = 1e-12;

    % Geodesic step along SPD manifold
    M_mid_pow = Vm * diag(dm .^ (1 - alpha)) * Vm';
    B_riem = G_sqrt * M_mid_pow * G_sqrt;

    % Re-scale trace to guarantee strict scale conservation
    tr_riem = trace(B_riem);
    if tr_riem > 0 && tr_B > 0
        B_riem = B_riem * (tr_B / tr_riem);
    end
    B_final = B_riem + gamma * eye(N_chan);

else
    % --- Trace-Matched Linear Convex Blending ---
    B_final = (1 - alpha) * B_sub + alpha * G_scaled + gamma * eye(N_chan);
end

% Ensure matrix is real and symmetric
B_final = real(B_final);
B_final = (B_final + B_final') / 2;

end
