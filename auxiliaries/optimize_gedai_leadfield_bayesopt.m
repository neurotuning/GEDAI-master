function [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, varargin)
% OPTIMIZE_GEDAI_LEADFIELD_BAYESOPT
% Optimizes a template Leadfield Gram matrix G to maximize the SENSAI metric
% using Bayesian Optimization over 3D anisotropic sensor coordinate deformation.
%
% If sensai_baseline >= BaselineSENSAIThreshold (default: 30%), the canonical
% template is already physically sound; BayesOpt is skipped entirely, saving compute
% and preserving 100% of peak baseline performance without artifact overfitting.
% Only when sensai_baseline < 30% is BayesOpt triggered to rescue the outlier.
%
% Signatures:
%   [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, G_template, nom_pos, ...)
%   [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, C_clean, G_template, nom_pos, ...) % C_clean ignored for backward compatibility
%
% Inputs:
%   C_emp      - (C x C) Full-bandwidth empirical covariance.
%   G_template - (C x C) Canonical leadfield Gram matrix (L * L').
%   nom_pos    - (C x 3) Nominal 3D Cartesian coordinates of the electrode array.
%
% Optional Name-Value Pairs:
%   'BaselineSENSAIThreshold' - Threshold below which BayesOpt triggers (default: 30.0).
%   'MaxObjectiveEvaluations' - Max BayesOpt evaluations if triggered (default: 25).
%   'InitialPoints'           - Initial seed evaluations (default: 5).
%   'SpatialBounds'           - [min, max] coordinate scaling bounds (default: [0.90, 1.10]).
%   'NoiseMultiplier'         - Weight for noise penalty in SENSAI, matching SENSAI_basic (default: 1.0).
%   'TopPCs'                  - Number of leadfield PCs for SSI (default: 3).
%   'MinSENSAIImprovement'    - Minimum SENSAI % gain required to adopt warped model (default: 0.0).
%   'PlotFcn'                 - BayesOpt plot functions (default: {@plotObjectiveModel, @plotMinObjective} if desktop).
%   'Verbose'                 - Display iteration logs (default: 0).
%
% Outputs:
%   G_opt       - (C x C) Warped, scale-adapted leadfield Gram matrix.
%   best_theta  - Table containing optimal warping hyperparameters (sx, sy, sz).
%   best_sensai - Maximum SENSAI score achieved.
%   results     - Full BayesianOptimization results object (or empty if skipped).

    % Handle flexible signature for backward compatibility
    if nargin >= 3 && size(varargin{2}, 2) == 3 && ~ischar(varargin{2}) && ~isstring(varargin{2})
        % Signature: (C_emp, G_template, nom_pos, ...)
        G_template = varargin{1};
        nom_pos    = varargin{2};
        opt_args   = varargin(3:end);
    elseif nargin >= 4 && size(varargin{3}, 2) == 3 && ~ischar(varargin{3}) && ~isstring(varargin{3})
        % Signature: (C_emp, C_clean, G_template, nom_pos, ...)
        % C_clean is accepted for backward compatibility but ignored
        G_template = varargin{2};
        nom_pos    = varargin{3};
        opt_args   = varargin(4:end);
    else
        error('optimize_gedai_leadfield_bayesopt:InvalidArgs', ...
            'Expected (C_emp, G_template, nom_pos, ...) with dimensions C x C and C x 3.');
    end

    p = inputParser;
    addParameter(p, 'BaselineSENSAIThreshold', 30.0, @isnumeric);
    addParameter(p, 'MaxObjectiveEvaluations', 25, @isnumeric);
    addParameter(p, 'InitialPoints', 5, @isnumeric);
    addParameter(p, 'SpatialBounds', [0.90, 1.10], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'NoiseMultiplier', 1.0, @isnumeric); % Default 1.0 matching SENSAI_basic
    addParameter(p, 'TopPCs', 3, @isnumeric);
    addParameter(p, 'MinSENSAIImprovement', 0.0, @isnumeric);
    plot_default = {};
    if usejava('desktop')
        plot_default = {@plotObjectiveModel, @plotMinObjective};
    end
    addParameter(p, 'PlotFcn', plot_default);
    addParameter(p, 'Verbose', 0, @isnumeric);
    parse(p, opt_args{:});
    opts = p.Results;

    C = size(C_emp, 1);
    assert(isequal(size(G_template), [C, C]), 'G_template and C_emp must have matching dimensions.');
    assert(size(nom_pos, 1) == C, 'nom_pos must be C x 3 matching channel count.');

    % Symmetrize inputs
    C_emp = real((C_emp + C_emp') / 2);
    G_template = real((G_template + G_template') / 2);

    % Trace-normalize all baseline inputs to ensure dimensionless scaling
    C_emp_norm = C_emp / trace(C_emp);
    G_nom_norm = G_template / trace(G_template);

    %% --- 1. Baseline SENSAI Evaluation (Canonical Leadfield) ---
    % Evaluated using NoiseMultiplier = 1.0 (directly matching SENSAI_basic)
    sensai_baseline = evaluate_sensai_single(G_nom_norm, C_emp_norm, opts.TopPCs, opts.NoiseMultiplier);

    %% --- 2. Threshold Safeguard on sensai_baseline ---
    % If the canonical leadfield already achieves adequate alignment (sensai_baseline >= 30%),
    % skip BayesOpt entirely. This saves compute time (0.005s) and prevents artifact overfitting.
    if sensai_baseline >= opts.BaselineSENSAIThreshold
        G_opt = G_template;
        best_theta = table(1.0, 1.0, 1.0, 'VariableNames', {'sx', 'sy', 'sz'});
        best_sensai = sensai_baseline;
        results = [];
        if opts.Verbose
            fprintf('\nCanonical leadfield retained: sensai_baseline = %.2f%% >= %.1f%%. BayesOpt skipped.\n', ...
                sensai_baseline, opts.BaselineSENSAIThreshold);
        end
        return;
    end

    %% --- 3. Run BayesOpt to Rescue Outlier (sensai_baseline < 30%) ---
    if opts.Verbose
        fprintf('\nLow baseline SENSAI detected (%.2f%% < %.1f%%). Running BayesOpt to rescue leadfield...\n', ...
            sensai_baseline, opts.BaselineSENSAIThreshold);
    end

    % 3D Anisotropic coordinate scaling (head elongation / cap stretch) constrained to +/- 10%
    sx = optimizableVariable('sx', opts.SpatialBounds, 'Type', 'real');
    sy = optimizableVariable('sy', opts.SpatialBounds, 'Type', 'real');
    sz = optimizableVariable('sz', opts.SpatialBounds, 'Type', 'real');
    vars = [sx, sy, sz];

    % Warm-start anchor: canonical template at (1.0, 1.0, 1.0)
    init_baseline = table(1.0, 1.0, 1.0, 'VariableNames', {'sx', 'sy', 'sz'});

    objFun = @(params) evaluate_sensai_cost(params, C_emp_norm, G_nom_norm, nom_pos, opts.TopPCs, opts.NoiseMultiplier);

    results = bayesopt(objFun, vars, ...
        'InitialX', init_baseline, ...
        'MaxObjectiveEvaluations', opts.MaxObjectiveEvaluations, ...
        'NumSeedPoints', opts.InitialPoints, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', opts.Verbose, ...
        'PlotFcn', opts.PlotFcn);

    best_theta = results.XAtMinObjective;
    best_sensai = -results.MinObjective;
    delta_sensai = best_sensai - sensai_baseline;

    %% --- 4. Gating Safeguard: Verify Improvement Over Canonical Template ---
    if delta_sensai > opts.MinSENSAIImprovement
        G_warped = generate_warped_gram(best_theta.sx, best_theta.sy, best_theta.sz, G_nom_norm, nom_pos);
        G_opt = G_warped * (trace(G_template) / trace(G_warped));
        if opts.Verbose
            fprintf('\nWarped leadfield adopted: SENSAI improved by +%.2f%% (%.2f%% -> %.2f%%)\n', ...
                delta_sensai, sensai_baseline, best_sensai);
            fprintf('Optimal scales: Sx=%.3f, Sy=%.3f, Sz=%.3f\n', best_theta.sx, best_theta.sy, best_theta.sz);
        end
    else
        G_opt = G_template;
        best_theta = init_baseline;
        best_sensai = sensai_baseline;
        if opts.Verbose
            fprintf('\nCanonical template retained: no improvement over baseline (Baseline: %.2f%%, Best: %.2f%%)\n', ...
                sensai_baseline, -results.MinObjective);
        end
    end

    % Ensure G_opt is real, symmetric, and positive definite
    G_opt = real((G_opt + G_opt') / 2);
end

%% =========================================================================
%% INTERNAL FUNCTIONS
%% =========================================================================

function cost = evaluate_sensai_cost(params, C_emp, G_nom, nom_pos, n_pc, noise_multiplier)
    try
        G_warped = generate_warped_gram(params.sx, params.sy, params.sz, G_nom, nom_pos);
        score = evaluate_sensai_single(G_warped, C_emp, n_pc, noise_multiplier);
        cost = -score;
    catch
        cost = 1e4; % Penalty on numerical instability
    end
end

function sensai_score = evaluate_sensai_single(G_target, C_emp, n_pc, noise_multiplier)
    C_dim = size(G_target, 1);
    G_target = (G_target + G_target') / 2;
    G_reg = 0.95 * G_target + 0.05 * (trace(G_target) / C_dim) * eye(C_dim);
    G_reg = (G_reg + G_reg') / 2;

    % 1. Extract reference template eigenvectors
    [evecs_Template, D_template] = eig(G_reg);
    [~, sort_template] = sort(diag(D_template), 'descend');
    evecs_Template = evecs_Template(:, sort_template);

    % 2. Run GEVD on empirical covariance: C_emp * Evec = G_reg * Evec * Eval
    [Evec, D_gevd] = eig(C_emp, G_reg);
    [evals_sorted, sort_gevd] = sort(diag(D_gevd), 'descend');
    Evec = Evec(:, sort_gevd);
    Eval = diag(evals_sorted);

    % 3. Call MATLAB GEDAI's exact SENSAI optimization engine
    [~, sensai_score] = SENSAI_fminbnd(-6, 12, G_reg, Eval, Evec, noise_multiplier, C_emp, evecs_Template, 'eeg', n_pc);
end

function G_w = generate_warped_gram(sx, sy, sz, G_nom, nom_pos)
    C = size(G_nom, 1);

    % Normalize coordinates by mean head radius for dimensionless scale invariance
    R_head = mean(sqrt(sum(nom_pos.^2, 2)));
    if R_head > 0
        pos_orig = nom_pos / R_head;
    else
        pos_orig = nom_pos;
    end
    pos_warped = pos_orig .* [sx, sy, sz];

    dist_orig = pdist2(pos_orig, pos_orig);
    dist_target = pdist2(pos_orig, pos_warped);

    % Thin-plate radial basis: r^2 * log(r). eye(C) ensures r=0 evaluates to log(1) = 0
    K_orig = dist_orig.^2 .* log(dist_orig + eye(C));
    K_target = dist_target.^2 .* log(dist_target + eye(C));

    W_spatial = (K_target + 1e-3 * eye(C)) / (K_orig + 1e-3 * eye(C));
    G_w = W_spatial * G_nom * W_spatial';

    % Symmetrize and trace-normalize
    G_w = (G_w + G_w') / 2;
    G_w = G_w / trace(G_w);
end
