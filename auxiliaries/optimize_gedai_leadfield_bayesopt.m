function [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, C_clean, G_template, nom_pos, varargin)
% OPTIMIZE_GEDAI_LEADFIELD_BAYESOPT
% Optimizes a template Leadfield Gram matrix G to maximize the SENSAI metric
% using Bayesian Optimization over spatial deformation, spectral tilt, and
% Riemannian geodesic shrinkage.
%
% Inputs:
%   C_emp      - (C x C) Full-bandwidth, artifact-containing empirical covariance.
%   C_clean    - (C x C) Robust / artifact-attenuated covariance (e.g. 8-25 Hz
%                band-passed or trimmed covariance) used for warping guidance.
%   G_template - (C x C) Canonical leadfield Gram matrix (L * L').
%   nom_pos    - (C x 3) Nominal 3D Cartesian coordinates of the electrode array.
%
% Optional Name-Value Pairs:
%   'MaxObjectiveEvaluations' - Max BayesOpt evaluations (default: 40).
%   'InitialPoints'           - Initial seed evaluations (default: 10).
%   'NoiseMultiplier'         - Weight for noise penalty in SENSAI (default: 3.0).
%   'TopPCs'                  - Number of leadfield PCs for SSI (default: 3).
%   'PlotFcn'                 - BayesOpt plot functions (default: {@plotObjectiveModel, @plotMinObjective}).
%   'Verbose'                 - Display iteration logs (default: 1).
%
% Outputs:
%   G_opt       - (C x C) Warped, scale-adapted leadfield Gram matrix.
%   best_theta  - Table containing optimal warping hyperparameters.
%   best_sensai - Maximum SENSAI score achieved.
%   results     - Full BayesianOptimization results object.

    p = inputParser;
    addRequired(p, 'C_emp', @(x) isnumeric(x) && ismatrix(x));
    addRequired(p, 'C_clean', @(x) isnumeric(x) && ismatrix(x));
    addRequired(p, 'G_template', @(x) isnumeric(x) && ismatrix(x));
    addRequired(p, 'nom_pos', @(x) isnumeric(x) && size(x, 2) == 3);
    addParameter(p, 'MaxObjectiveEvaluations', 40, @isnumeric);
    addParameter(p, 'InitialPoints', 10, @isnumeric);
    addParameter(p, 'NoiseMultiplier', 3.0, @isnumeric);
    addParameter(p, 'TopPCs', 3, @isnumeric);
    addParameter(p, 'PlotFcn', {@plotObjectiveModel, @plotMinObjective});
    addParameter(p, 'Verbose', 1, @isnumeric);
    parse(p, C_emp, C_clean, G_template, nom_pos, varargin{:});
    opts = p.Results;

    C = size(C_emp, 1);
    assert(isequal(size(C_clean), [C, C]), 'C_clean and C_emp must have matching dimensions.');
    assert(isequal(size(G_template), [C, C]), 'G_template and C_emp must have matching dimensions.');
    assert(size(nom_pos, 1) == C, 'nom_pos must be C x 3 matching channel count.');

    % Symmetrize and regularize inputs
    C_emp = real((C_emp + C_emp') / 2);
    C_clean = real((C_clean + C_clean') / 2);
    G_template = real((G_template + G_template') / 2);

    % Trace-normalize all baseline inputs to ensure dimensionless scaling
    C_emp_norm = C_emp / trace(C_emp);
    C_clean_norm = C_clean / trace(C_clean);
    G_nom_norm = G_template / trace(G_template);

    %% --- 1. Define Search Space for BayesOpt ---
    % 3D Anisotropic coordinate scaling (head elongation / cap stretch)
    sx = optimizableVariable('sx', [0.85, 1.15], 'Type', 'real');
    sy = optimizableVariable('sy', [0.85, 1.15], 'Type', 'real');
    sz = optimizableVariable('sz', [0.85, 1.15], 'Type', 'real');

    % Global spectral eigenvalue tilt (skull conductivity / attenuation factor)
    tau = optimizableVariable('tau', [0.70, 1.30], 'Type', 'real');

    % Manifold geodesic shrinkage toward C_clean (constrained to prevent overfit)
    t_shrink = optimizableVariable('t_shrink', [0.0, 0.20], 'Type', 'real');

    vars = [sx, sy, sz, tau, t_shrink];

    %% --- 2. Objective Function for BayesOpt ---
    objFun = @(params) evaluate_sensai_cost(params, C_emp_norm, C_clean_norm, G_nom_norm, nom_pos, opts.TopPCs, opts.NoiseMultiplier);

    %% --- 3. Run Bayesian Optimization ---
    if opts.Verbose
        fprintf('\nRunning BayesOpt over leadfield warping parameters to maximize SENSAI...\n');
    end

    results = bayesopt(objFun, vars, ...
        'MaxObjectiveEvaluations', opts.MaxObjectiveEvaluations, ...
        'NumSeedPoints', opts.InitialPoints, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', opts.Verbose, ...
        'PlotFcn', opts.PlotFcn);

    best_theta = results.XAtMinObjective;
    best_sensai = -results.MinObjective;

    % Reconstruct optimal leadfield Gram matrix using best hyperparameters
    G_opt = generate_warped_gram(best_theta, G_nom_norm, nom_pos, C_clean_norm);

    % Rescale back to match original nominal power
    G_opt = G_opt * (trace(G_template) / trace(G_opt));

    if opts.Verbose
        fprintf('\n==========================================\n');
        fprintf('Optimal SENSAI Score: %.4f\n', best_sensai);
        fprintf('Scale factors: Sx=%.3f, Sy=%.3f, Sz=%.3f\n', best_theta.sx, best_theta.sy, best_theta.sz);
        fprintf('Spectral Tilt (tau): %.3f | Geodesic Step (t): %.3f\n', best_theta.tau, best_theta.t_shrink);
        fprintf('==========================================\n');
    end
end

%% =========================================================================
%% INTERNAL FUNCTIONS
%% =========================================================================

function cost = evaluate_sensai_cost(params, C_emp, C_clean, G_template, nom_pos, n_pc, noise_multiplier)
    try
        % 1. Warp Gram matrix using candidate hyperparameters
        G_warped = generate_warped_gram(params, G_template, nom_pos, C_clean);

        % Symmetrize and add small regularizer to guarantee positive definiteness
        C_dim = size(G_warped, 1);
        G_warped = (G_warped + G_warped') / 2;
        G_reg = 0.95 * G_warped + 0.05 * (trace(G_warped) / C_dim) * eye(C_dim);
        G_reg = (G_reg + G_reg') / 2;

        % 2. Extract reference template eigenvectors from adapted leadfield model
        [evecs_Template, D_template] = eig(G_reg);
        [~, sort_template] = sort(diag(D_template), 'descend');
        evecs_Template = evecs_Template(:, sort_template);

        % 3. Run GEVD on empirical covariance: C_emp * Evec = G_reg * Evec * Eval
        [Evec, D_gevd] = eig(C_emp, G_reg);
        [evals_sorted, sort_gevd] = sort(diag(D_gevd), 'descend');
        Evec = Evec(:, sort_gevd);
        Eval = diag(evals_sorted);

        % 4. Call MATLAB GEDAI's exact SENSAI optimization engine
        % Uses the calibrated 2-step power subspace iteration and log-eigenvalue scaling
        [~, maxSENSAIScore] = SENSAI_fminbnd(-6, 12, G_reg, Eval, Evec, noise_multiplier, C_emp, evecs_Template, 'eeg', n_pc);

        % Minimize negative SENSAI (cost in percentage, e.g. -65.4%)
        cost = -maxSENSAIScore;
    catch
        cost = 1e4; % Penalty on numerical instability
    end
end

function G_w = generate_warped_gram(params, G_nom, nom_pos, C_clean)
    C = size(G_nom, 1);

    % A. 3D Spline Deformation from Nominal Sensor Stretch
    % Normalizing coordinates by head radius ensures scale invariance (meters vs mm)
    R_head = mean(sqrt(sum(nom_pos.^2, 2)));
    if R_head > 0
        pos_orig = nom_pos / R_head;
    else
        pos_orig = nom_pos;
    end
    pos_warped = pos_orig .* [params.sx, params.sy, params.sz];

    dist_orig = pdist2(pos_orig, pos_orig);
    dist_target = pdist2(pos_orig, pos_warped);

    % Thin-plate radial basis: r^2 * log(r). eye(C) ensures r=0 evaluates to log(1) = 0
    K_orig = dist_orig.^2 .* log(dist_orig + eye(C));
    K_target = dist_target.^2 .* log(dist_target + eye(C));

    W_spatial = (K_target + 1e-3 * eye(C)) / (K_orig + 1e-3 * eye(C));
    G_w = W_spatial * G_nom * W_spatial';

    % B. Spectral Eigenvalue Tilt (tau) on G
    [Vg, Dg] = eig((G_w + G_w') / 2);
    [dg_vals, dg_idx] = sort(diag(Dg), 'descend');
    Vg = Vg(:, dg_idx);
    dg = max(dg_vals, 1e-12);
    G_w = Vg * diag(dg.^params.tau) * Vg';

    % C. Geodesic Pull Guided by C_clean (Riemannian geometry on SPD manifold)
    if params.t_shrink > 0
        G_inv_sqrt = Vg * diag(1 ./ sqrt(dg)) * Vg';
        G_sqrt = Vg * diag(sqrt(dg)) * Vg';

        M_inner = G_inv_sqrt * C_clean * G_inv_sqrt;
        M_inner = (M_inner + M_inner') / 2;

        [Vm, Dm] = eig(M_inner);
        dm = max(diag(Dm), 1e-12);

        % Fractional power geodesic step: G^(1/2) * (G^(-1/2) * C_clean * G^(-1/2))^t * G^(1/2)
        G_w = G_sqrt * (Vm * diag(dm.^params.t_shrink) * Vm') * G_sqrt;
    end

    % Symmetrize and trace-normalize
    G_w = (G_w + G_w') / 2;
    G_w = G_w / trace(G_w);
end
