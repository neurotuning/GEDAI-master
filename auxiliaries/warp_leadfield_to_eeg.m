function [G_warped, diag_info] = warp_leadfield_to_eeg(epochs, G, k_clean, k_ecs, blend_mode, srate, epoch_size)
% WARP_LEADFIELD_TO_EEG Warps leadfield Gram matrix toward empirical EEG subspace
% while preserving full rank and spectral decay via a unitary rotation U in SO(n).
%
% Usage:
%   [G_warped, diag_info] = warp_leadfield_to_eeg(epochs, G)
%   [G_warped, diag_info] = warp_leadfield_to_eeg(epochs, G, k_clean, k_ecs, blend_mode)
%   [G_warped, diag_info] = warp_leadfield_to_eeg(epochs, G, k_clean, k_ecs, blend_mode, srate, epoch_size)
%
% Inputs:
%   epochs     - (n_chan x n_samples x n_epochs) EEG array, 2D continuous array, or EEGLAB struct
%   G          - (n_chan x n_chan) Leadfield Gram/covariance matrix
%   k_clean    - Number of cleanest GEVD components per epoch (e.g., 3-8). Default: 3
%   k_ecs      - Dimension of empirical common subspace (e.g., 8-16). Default: k_clean
%   blend_mode - 'adaptive' (default) or scalar value in [0, 1]
%   srate      - (Optional) Sampling rate in Hz for continuous windowing. Default: 250
%   epoch_size - (Optional) Window size in seconds for continuous windowing. Default: 1
%
% Outputs:
%   G_warped   - Full-rank warped leadfield covariance matrix (U * G * U')
%   diag_info  - Struct containing principal angles, t-weights, rotation matrix, etc.

    if nargin < 5 || isempty(blend_mode), blend_mode = 'adaptive'; end
    if nargin < 4 || isempty(k_ecs),      k_ecs = k_clean; end
    if nargin < 3 || isempty(k_clean),    k_clean = 3; end
    if nargin < 6 || isempty(srate),      srate = 250; end
    if nargin < 7 || isempty(epoch_size), epoch_size = 1; end

    if nargin < 2 || isempty(G)
        error('Leadfield Gram matrix G is required.');
    end

    % --- Data Extraction & Windowing ---
    if isstruct(epochs) && isfield(epochs, 'data')
        if isfield(epochs, 'srate') && ~isempty(epochs.srate)
            srate = epochs.srate;
        end
        data = double(epochs.data);
    else
        data = double(epochs);
    end

    [n_chan, n_pts, n_epochs] = size(data);

    % Automatic windowing for continuous 2D data
    if n_epochs == 1 && n_pts >= round(srate * epoch_size)
        epoch_samples = round(srate * epoch_size);
        num_epochs = floor(n_pts / epoch_samples);
        if num_epochs >= 1
            data = reshape(data(:, 1:num_epochs * epoch_samples), n_chan, epoch_samples, num_epochs);
            [n_chan, ~, n_epochs] = size(data);
        end
    end

    k_clean = min(k_clean, n_chan);
    k_ecs   = min(k_ecs, n_chan);

    % Ensure G is symmetric
    G = real((G + G') / 2);

    % --- 1. Extract Clean Spatial Patterns via GEVD ---
    reg_val_G = trace(G) / n_chan;
    G_reg = G + 1e-6 * max(reg_val_G, 1e-12) * eye(n_chan);
    G_reg = (G_reg + G_reg') / 2;
    U_cells = cell(n_epochs, 1);

    for e = 1:n_epochs
        if n_epochs == 1
            X_e = data;
        else
            X_e = data(:, :, e);
        end
        
        X_e = X_e - mean(X_e, 2);
        n_samples_e = size(X_e, 2);
        C_e = (X_e * X_e') / max(1, n_samples_e - 1);
        C_e = real((C_e + C_e') / 2);
        
        [W_e, D_e] = eig(C_e, G_reg);
        eigvals_e = real(diag(D_e));
        [~, sort_idx] = sort(eigvals_e, 'ascend'); % Smallest eigenvalues = cleanest
        
        take_k = min(k_clean, length(sort_idx));
        W_clean = W_e(:, sort_idx(1:take_k));
        
        denom = W_clean' * C_e * W_clean;
        denom_reg = denom + 1e-12 * eye(size(denom));
        try
            A_clean = (C_e * W_clean) / denom_reg;
        catch
            A_clean = (C_e * W_clean) * pinv(denom);
        end
        
        [U_cells{e}, ~] = qr(A_clean, 0);
    end

    % --- 2. Extract Empirical Common Subspace (ECS) ---
    [V_ecs, outlier_info] = extract_common_subspace(U_cells, k_ecs, 0.45);

    % --- 3. Principal Angles between Leadfield and ECS ---
    [V_L_all, D_L] = eig((G + G') / 2);
    [~, sort_L] = sort(real(diag(D_L)), 'descend');
    V_L_all = V_L_all(:, sort_L);
    
    k_actual = min([k_ecs, size(V_L_all, 2), size(V_ecs, 2)]);
    V_L = V_L_all(:, 1:k_actual);
    V_ecs = V_ecs(:, 1:k_actual);

    % SVD on mutual projection to find canonical coordinate bases
    [Y, S_mat, Z] = svd(V_L' * V_ecs);
    cos_theta = min(max(diag(S_mat), -1), 1);
    theta_rad = acos(cos_theta);
    theta_deg = rad2deg(theta_rad);

    % --- 4. Angle-Dependent Weight Vector t ---
    if ischar(blend_mode) && strcmpi(blend_mode, 'adaptive')
        t_vec = compute_adaptive_t(theta_deg);
    elseif isnumeric(blend_mode)
        if isscalar(blend_mode)
            t_vec = repmat(blend_mode, k_actual, 1);
        else
            t_vec = blend_mode(:);
            if length(t_vec) < k_actual
                t_vec = [t_vec; repmat(t_vec(end), k_actual - length(t_vec), 1)];
            else
                t_vec = t_vec(1:k_actual);
            end
        end
    else
        t_vec = compute_adaptive_t(theta_deg);
    end

    % --- 5. Construct Unitary Subspace Rotation Matrix U in SO(n) ---
    % Find orthogonal tangent basis R spanning the deviation from V_L to V_ecs
    proj_diff = (eye(n_chan) - V_L * V_L') * (V_ecs * Z);
    [R, ~] = qr(proj_diff, 0);

    % Build the block Givens generator across the 2k principal rotation planes
    % U = I + K * M_rot * K', where K = [V_L*Y, R] is an orthonormal (n x 2k) basis
    K = [V_L * Y, R];
    
    phi = t_vec .* theta_rad;
    C_diag = diag(cos(phi) - 1);
    S_diag = diag(sin(phi));
    
    M_rot = [C_diag, -S_diag; ...
             S_diag,  C_diag];
         
    U_rot = eye(n_chan) + K * M_rot * K';

    % --- 6. Apply Full-Rank Congruence Transformation ---
    % Rotates the dominant subspace toward V_ecs while leaving the orthogonal
    % complement and the full eigenvalue spectrum of G completely intact.
    G_warped = U_rot * G * U_rot';
    G_warped = real((G_warped + G_warped') / 2);

    % Diagnostics
    diag_info.principal_angles_deg = theta_deg;
    diag_info.principal_angles_rad = theta_rad;
    diag_info.t_weights            = t_vec;
    diag_info.outlier_info         = outlier_info;
    diag_info.U_rot                = U_rot;
    diag_info.V_L                  = V_L;
    diag_info.V_ecs                = V_ecs;
    diag_info.singular_values      = diag(S_mat);

    % =========================================================================
    % NESTED FUNCTIONS
    % =========================================================================

    function [V_common, out_info] = extract_common_subspace(PC_cells, k_common, outlier_thresh)
        if nargin < 3 || isempty(outlier_thresh), outlier_thresh = 0.45; end

        M = numel(PC_cells);
        U_concat = [];
        P_sum = zeros(n_chan, n_chan);
        pc_map = [];

        for m = 1:M
            U_m = PC_cells{m};
            [U_m, ~] = qr(U_m, 0);
            
            U_concat = [U_concat, U_m];
            P_sum = P_sum + (U_m * U_m');
            
            k_m = size(U_m, 2);
            pc_map = [pc_map; [repmat(m, k_m, 1), (1:k_m)']];
        end

        [V_mat, D_mat] = eig((P_sum + P_sum') / 2);
        [eigvals, sort_order] = sort(real(diag(D_mat)), 'descend');
        V_mat = V_mat(:, sort_order);

        if nargin < 2 || isempty(k_common)
            k_common = sum(eigvals > (0.5 * M));
            k_common = max(k_common, 1);
        end
        k_common = min(k_common, size(V_mat, 2));

        V_common = V_mat(:, 1:k_common);
        P_common = V_common * V_common';

        total_pcs = size(U_concat, 2);
        alignment_scores = zeros(total_pcs, 1);
        outlier_scores   = zeros(total_pcs, 1);

        for i = 1:total_pcs
            u = U_concat(:, i);
            alignment_scores(i) = norm(V_common' * u)^2;
            outlier_scores(i)   = 1 - alignment_scores(i);
        end

        epoch_distances = zeros(M, 1);
        for m = 1:M
            U_m = PC_cells{m};
            P_m = U_m * U_m';
            epoch_distances(m) = norm(P_m - P_common, 'fro') / sqrt(2);
        end

        out_info.pc_epoch_idx     = pc_map(:, 1);
        out_info.pc_index         = pc_map(:, 2);
        out_info.alignment_scores = alignment_scores;
        out_info.outlier_scores   = outlier_scores;
        out_info.is_outlier       = outlier_scores > outlier_thresh;
        out_info.epoch_distances  = epoch_distances;
        out_info.singular_values  = eigvals;
    end

    function t_out = compute_adaptive_t(theta_degrees)
        t_out = zeros(size(theta_degrees));
        for i = 1:numel(theta_degrees)
            th = theta_degrees(i);
            if th <= 10
                t_out(i) = 0.1 * (th / 10);
            elseif th > 10 && th <= 40
                t_out(i) = 0.1 + 0.65 * sin(((th - 10) / 30) * (pi / 2));
            elseif th > 40 && th <= 75
                t_out(i) = 0.75 * cos(((th - 40) / 35) * (pi / 2));
            else
                t_out(i) = 0.0; % Reject unmodeled artifact modes (> 75 deg)
            end
        end
    end

end
