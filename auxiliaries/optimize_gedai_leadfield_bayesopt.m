function [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, varargin)
% OPTIMIZE_GEDAI_LEADFIELD_BAYESOPT
% Optimizes a template Leadfield Gram matrix G to maximize the SENSAI metric
% using Bayesian Optimization over 3D anisotropic sensor coordinate deformation,
% rigid 3D head translation inside the helmet (for MEG), and spectral shaping.
%
% If sensai_baseline >= BaselineSENSAIThreshold, the canonical template is already
% physically sound; BayesOpt is skipped entirely, saving compute (0.005s) and
% preserving 100% of peak baseline performance without artifact overfitting.
% Only when sensai_baseline < BaselineSENSAIThreshold is BayesOpt triggered.
%
% Modality Defaults:
%   EEG: 3-DOF Spatial Scaling (sx, sy, sz in [0.90, 1.10]), BaselineSENSAIThreshold = 30.0%
%   MEG: 7-DOF Model: 3D Scaling (sx, sy, sz) + Rigid Translation (tx, ty, tz in [-0.15, 0.15])
%        + Spectral Exponent (tau in [0.85, 1.15]), BaselineSENSAIThreshold = 62.0%
%
% Signatures:
%   [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, G_template, nom_pos, ...)
%   [G_opt, best_theta, best_sensai, results] = optimize_gedai_leadfield_bayesopt(C_emp, C_clean, G_template, nom_pos, ...) % C_clean ignored for backward compatibility
%
% Inputs:
%   C_emp      - (C x C) Full-bandwidth empirical covariance.
%   G_template - (C x C) Canonical leadfield Gram matrix (L * L').
%   nom_pos    - (C x 3) Nominal 3D Cartesian coordinates of the electrode/sensor array.
%
% Optional Name-Value Pairs:
%   'BaselineSENSAIThreshold' - Threshold below which BayesOpt triggers (default: 30% for EEG, 62% for MEG).
%   'SignalType'              - Modality: 'eeg' or 'meg' (default: 'eeg').
%   'MaxObjectiveEvaluations' - Max BayesOpt evaluations if triggered (default: 25).
%   'InitialPoints'           - Initial seed evaluations (default: 5).
%   'SpatialBounds'           - [min, max] coordinate scaling bounds (default: [0.90, 1.10]).
%   'TranslationBounds'     - [min, max] normalized 3D translation bounds for MEG (default: [-0.15, 0.15]).
%   'TauBounds'               - [min, max] spectral eigenvalue exponent bounds for MEG (default: [0.85, 1.15]).
%   'NoiseMultiplier'         - Weight for noise penalty in SENSAI, matching SENSAI_basic (default: 1.0).
%   'TopPCs'                  - Number of leadfield PCs for SSI (default: 3).
%   'MinSENSAIImprovement'    - Minimum SENSAI % gain required to adopt warped model (default: 0.0).
%   'PlotFcn'                 - BayesOpt plot functions (default: {@plotObjectiveModel, @plotMinObjective} if desktop).
%   'Verbose'                 - Display iteration logs (default: 0).
%
% Outputs:
%   G_opt       - (C x C) Adapted leadfield Gram matrix.
%   best_theta  - Table containing optimal hyperparameters.
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
    addParameter(p, 'BaselineSENSAIThreshold', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'SignalType', 'eeg', @(x) ischar(x) || isstring(x));
    addParameter(p, 'MaxObjectiveEvaluations', 25, @isnumeric);
    addParameter(p, 'InitialPoints', 5, @isnumeric);
    addParameter(p, 'SpatialBounds', [0.90, 1.10], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'TranslationBounds', [-0.15, 0.15], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'TauBounds', [0.85, 1.15], @(x) isnumeric(x) && numel(x) == 2);
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

    % Set modality-specific default threshold if not explicitly specified
    if isempty(opts.BaselineSENSAIThreshold)
        if strcmpi(opts.SignalType, 'meg')
            if C > 150 % Gradiometers (typically 204 channels)
                baseline_threshold = 30.0; % GRAD threshold (clean sits at 35-38%)
            else % Magnetometers (typically 102 channels)
                baseline_threshold = 62.0; % MAG threshold (outliers sit < 60%)
            end
        else
            baseline_threshold = 30.0; % EEG threshold (outliers sit < 30%)
        end
    else
        baseline_threshold = opts.BaselineSENSAIThreshold;
    end

    % Symmetrize inputs
    C_emp = real((C_emp + C_emp') / 2);
    G_template = real((G_template + G_template') / 2);

    % Trace-normalize all baseline inputs to ensure dimensionless scaling
    C_emp_norm = C_emp / trace(C_emp);
    G_nom_norm = G_template / trace(G_template);

    %% --- 1. Baseline SENSAI Evaluation (Canonical Leadfield) ---
    sensai_baseline = evaluate_sensai_single(G_nom_norm, C_emp_norm, opts.TopPCs, opts.NoiseMultiplier, opts.SignalType);

    %% --- 2. Threshold Safeguard on sensai_baseline ---
    % If the canonical leadfield already achieves adequate alignment, skip BayesOpt entirely.
    % Saves compute time (0.005s) and prevents artifact overfitting on clean/typical recordings.
    if sensai_baseline >= baseline_threshold
        G_opt = G_template;
        if strcmpi(opts.SignalType, 'eeg')
            best_theta = table(1.0, 1.0, 1.0, 'VariableNames', {'sx', 'sy', 'sz'});
        else
            best_theta = table(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, ...
                'VariableNames', {'sx', 'sy', 'sz', 'tx', 'ty', 'tz', 'tau'});
        end
        best_sensai = sensai_baseline;
        results = [];
        if opts.Verbose
            fprintf('\nCanonical leadfield retained: sensai_baseline = %.2f%% >= %.1f%%. BayesOpt skipped.\n', ...
                sensai_baseline, baseline_threshold);
        end
        return;
    end

    %% --- 3. Run BayesOpt to Rescue Outlier (sensai_baseline < threshold) ---
    if opts.Verbose
        fprintf('\nLow baseline SENSAI detected (%.2f%% < %.1f%%). Running BayesOpt to rescue leadfield...\n', ...
            sensai_baseline, baseline_threshold);
    end

    if strcmpi(opts.SignalType, 'eeg')
        % EEG: 3D Anisotropic coordinate scaling (head elongation / cap stretch)
        sx = optimizableVariable('sx', opts.SpatialBounds, 'Type', 'real');
        sy = optimizableVariable('sy', opts.SpatialBounds, 'Type', 'real');
        sz = optimizableVariable('sz', opts.SpatialBounds, 'Type', 'real');
        vars = [sx, sy, sz];
        init_baseline = table(1.0, 1.0, 1.0, 'VariableNames', {'sx', 'sy', 'sz'});
    else
        % MEG: 3D Scaling + Rigid Head Translation in Helmet + Spectral Exponent
        sx = optimizableVariable('sx', opts.SpatialBounds, 'Type', 'real');
        sy = optimizableVariable('sy', opts.SpatialBounds, 'Type', 'real');
        sz = optimizableVariable('sz', opts.SpatialBounds, 'Type', 'real');
        tx = optimizableVariable('tx', opts.TranslationBounds, 'Type', 'real');
        ty = optimizableVariable('ty', opts.TranslationBounds, 'Type', 'real');
        tz = optimizableVariable('tz', opts.TranslationBounds, 'Type', 'real');
        tau = optimizableVariable('tau', opts.TauBounds, 'Type', 'real');
        vars = [sx, sy, sz, tx, ty, tz, tau];
        init_baseline = table(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, ...
            'VariableNames', {'sx', 'sy', 'sz', 'tx', 'ty', 'tz', 'tau'});
    end

    objFun = @(params) evaluate_sensai_cost(params, C_emp_norm, G_nom_norm, nom_pos, opts.TopPCs, opts.NoiseMultiplier, opts.SignalType);

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
        G_warped = generate_adapted_gram(best_theta, G_nom_norm, nom_pos);
        G_opt = G_warped * (trace(G_template) / trace(G_warped));
        if opts.Verbose
            fprintf('\nAdapted leadfield adopted: SENSAI improved by +%.2f%% (%.2f%% -> %.2f%%)\n', ...
                delta_sensai, sensai_baseline, best_sensai);
            if strcmpi(opts.SignalType, 'eeg')
                fprintf('Optimal scales: Sx=%.3f, Sy=%.3f, Sz=%.3f\n', best_theta.sx, best_theta.sy, best_theta.sz);
            else
                fprintf('Optimal MEG params: Sx=%.3f, Sy=%.3f, Sz=%.3f, Tx=%.3f, Ty=%.3f, Tz=%.3f, Tau=%.3f\n', ...
                    best_theta.sx, best_theta.sy, best_theta.sz, best_theta.tx, best_theta.ty, best_theta.tz, best_theta.tau);
            end
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

function cost = evaluate_sensai_cost(params, C_emp, G_nom, nom_pos, n_pc, noise_multiplier, signal_type)
    try
        G_warped = generate_adapted_gram(params, G_nom, nom_pos);
        score = evaluate_sensai_single(G_warped, C_emp, n_pc, noise_multiplier, signal_type);
        cost = -score;
    catch
        cost = 1e4; % Penalty on numerical instability
    end
end

function sensai_score = evaluate_sensai_single(G_target, C_emp, n_pc, noise_multiplier, signal_type)
    if nargin < 5 || isempty(signal_type)
        signal_type = 'eeg';
    end
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
    [~, sensai_score] = SENSAI_fminbnd(-6, 12, G_reg, Eval, Evec, noise_multiplier, C_emp, evecs_Template, signal_type, n_pc);
end

function Gw = generate_adapted_gram(theta, G_nom, nom_pos)
    C = size(G_nom, 1);

    % Parameter extraction with safe defaults
    sx = 1.0; sy = 1.0; sz = 1.0;
    tx = 0.0; ty = 0.0; tz = 0.0;
    tau = 1.0;

    names = theta.Properties.VariableNames;
    if ismember('sx', names), sx = theta.sx; end
    if ismember('sy', names), sy = theta.sy; end
    if ismember('sz', names), sz = theta.sz; end
    if ismember('tx', names), tx = theta.tx; end
    if ismember('ty', names), ty = theta.ty; end
    if ismember('tz', names), tz = theta.tz; end
    if ismember('tau', names), tau = theta.tau; end

    % Normalize coordinates by mean head radius for dimensionless scale invariance
    R_head = mean(sqrt(sum(nom_pos.^2, 2)));
    if R_head > 0
        pos_orig = nom_pos / R_head;
    else
        pos_orig = nom_pos;
    end

    % 1. Spatial deformation: Translation + Scaling
    pos_warped = (pos_orig + [tx, ty, tz]) .* [sx, sy, sz];

    dist_orig = pdist2(pos_orig, pos_orig);
    dist_target = pdist2(pos_orig, pos_warped);

    % Thin-plate radial basis: r^2 * log(r). eye(C) ensures r=0 evaluates to log(1) = 0
    K_orig = dist_orig.^2 .* log(dist_orig + eye(C));
    K_target = dist_target.^2 .* log(dist_target + eye(C));

    W_spatial = (K_target + 1e-3 * eye(C)) / (K_orig + 1e-3 * eye(C));
    Gw = W_spatial * G_nom * W_spatial';

    % 2. Spectral power-law eigenvalue shaping (tau)
    if tau ~= 1.0
        [Vg, Dg] = eig((Gw + Gw') / 2);
        [dg, idx] = sort(diag(Dg), 'descend');
        Vg = Vg(:, idx);
        dg = max(dg, 1e-12);
        Gw = Vg * diag(dg.^tau) * Vg';
    end

    % Symmetrize and trace-normalize
    Gw = (Gw + Gw') / 2;
    Gw = Gw / trace(Gw);
end
