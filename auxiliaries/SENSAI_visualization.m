function metrics = SENSAI_visualization(EEGavRef, EEGclean, EEGartifacts, ref_cov, epoch_duration_sec, signal_type, SSI_top_PCs)
% SENSAI_VISUALIZATION  2D SENSAI scatter: subspace similarity vs epoch power
%
% Inputs:
%   EEGavRef           - Original EEG dataset BEFORE denoising
%   EEGclean           - Cleaned EEG dataset AFTER denoising (signal)
%   EEGartifacts       - EEG dataset containing only the removed artifacts
%   ref_cov            - Leadfield covariance matrix (Channels x Channels)
%   epoch_duration_sec - Duration (in seconds) of the epochs to be used for calculation (default 1s)
%   signal_type        - 'eeg' or 'meg'. Dictates the number of principal components.
%   SSI_top_PCs        - Optional: explicit number of principal components to use
%
% Axes:
%   X  –  SSI composite: geometric mean of top-N PC cosines  →  scalar in [0,1]
%          (1 = epoch subspace perfectly aligned with leadfield)
%   Y  –  log10(epoch power) = log10( trace(C) )
%
% Layout (side-by-side 2D scatters):
%   Left  : Before Denoising  (coloured by SSI value)
%   Right : After  Denoising  (green = signal, red = noise)
%           + 95% confidence ellipses + 2D LDA accuracy

%% ── 0. Parse inputs and parameters ──────────────────────────────────────
if nargin < 5 || isempty(epoch_duration_sec)
    epoch_duration_sec = 1; % 1-second window by default
end
if nargin < 6 || isempty(signal_type)
    signal_type = 'eeg';
end
if nargin < 7 || isempty(SSI_top_PCs)
    if strcmpi(signal_type, 'meg') || size(ref_cov, 1) <= 100
        SSI_top_PCs = 4; % MEG or sparse EEG
    else
        SSI_top_PCs = 3; % Typical EEG dense array
    end
end

%% ── 1. Epoch Data & Extract Covariances (Contiguous) ─────────────────
srate = EEGavRef.srate;
epoch_samples = round(srate * epoch_duration_sec);
epoch_step    = epoch_samples;

% Function to pad data and compute covariance cell arrays
function COV_array = compute_cov_array(data)
    pnts = size(data, 2);
    remainder = rem(pnts, epoch_samples);
    
    if remainder ~= 0
        samples_to_pad = epoch_samples - remainder;
        reflection_segment = data(:, end-samples_to_pad+1:end);
        padding = fliplr(reflection_segment); % Flip the segment left-to-right
        data = [data, padding];
    end
    
    num_epochs = size(data, 2) / epoch_samples;
    COV_array = cell(num_epochs, 1);
    for epo = 1:num_epochs
        i_start = (epo - 1) * epoch_samples + 1;
        i_end   = i_start + epoch_samples - 1;
        COV_array{epo} = cov(data(:, i_start:i_end)');
    end
end

C_before    = compute_cov_array(EEGavRef.data);
C_after     = compute_cov_array(EEGclean.data);
C_artifacts = compute_cov_array(EEGartifacts.data);

% Align BEFORE array to the same epoch count for a fair comparison (in case of differing lengths e.g. after epoch rejection)
num_epochs_after = length(C_after);
if length(C_before) > num_epochs_after
    C_before = C_before(1:num_epochs_after);
elseif length(C_before) < num_epochs_after
    C_after     = C_after(1:length(C_before));
    C_artifacts = C_artifacts(1:length(C_before));
end

%% ── 1. Principal-angle subspaces ────────────────────────────────────────
if nargin < 5 || isempty(SSI_top_PCs)
    if size(ref_cov, 1) > 100 % Typical EEG dense array
        SSI_top_PCs = 3;
    else
        SSI_top_PCs = 4; % MEG or sparse EEG
    end
end
[Vref, Dref] = eig(ref_cov);
[~, idx]     = sort(diag(Dref), 'descend');
basis_ref    = Vref(:, idx(1:SSI_top_PCs));

angs_before    = extract_angles(C_before,    basis_ref, SSI_top_PCs);
angs_after     = extract_angles(C_after,     basis_ref, SSI_top_PCs);
angs_artifacts = extract_angles(C_artifacts, basis_ref, SSI_top_PCs);

%% ── 2. Composite SSI  (geometric mean of 3 cosines → [0,1]) ────────────
ssi_before    = prod(angs_before,    2) .^ (1/SSI_top_PCs);
ssi_after     = prod(angs_after,     2) .^ (1/SSI_top_PCs);
ssi_artifacts = prod(angs_artifacts, 2) .^ (1/SSI_top_PCs);

%% ── 3. Epoch power  (log10 of trace) ────────────────────────────────────
lpow_before    = 10 * log10(extract_power(C_before));
lpow_after     = 10 * log10(extract_power(C_after));
lpow_artifacts = 10 * log10(extract_power(C_artifacts));

% Ideal Target: 100% Subspace Alignment at current signal power
ideal_power_target = median(lpow_after);

%% ── 4. 2D LDA on [SSI, log-power] ──────────────────────────────────────
X_lda = [ssi_after,     lpow_after; ...
         ssi_artifacts, lpow_artifacts];
Y_lda = [ones(numel(ssi_after), 1); zeros(numel(ssi_artifacts), 1)];
try
    lda_mdl      = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5);
    lda_accuracy = (1 - kfoldLoss(lda_mdl)) * 100;
catch
    lda_accuracy = NaN;
end

%% ── 5. Plot ──────────────────────────────────────────────────────────────
figure('Name', 'GEDAI SENSAI Analysis', 'Color', 'w', ...
       'Position', [80 100 1200 520]);

% Colour palette
col_sig  = [0.08 0.72 0.22];
col_noise= [0.85 0.13 0.13];
col_bef  = [0.30 0.45 0.75];
col_star = [1.00 0.88 0.00];

% ── Panel 1: Before GEDAI ────────────────────────────────────────────────
ax1 = subplot(1, 2, 1);
hold(ax1, 'on');

% Sort so high-SSI points render on top
[~, si] = sort(ssi_before, 'ascend');
scatter(ax1, lpow_before(si), ssi_before(si), 38, ssi_before(si), ...
        'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);

% Ideal alignment horizon
yline(ax1, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
draw_ellipse(ax1, lpow_before, ssi_before, col_bef, 0.95);

colormap(ax1, parula);
cb = colorbar(ax1, 'eastoutside');
cb.Label.String = 'SSI composite';  cb.Label.FontSize = 10;
clim(ax1, [0 1]);

xlabel(ax1, 'Epoch Power (dB)',                'FontSize', 11);
ylabel(ax1, sprintf('SSI  (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
title(ax1, sprintf('Before Denoising\nn = %d epochs  |  Mean SSI = %.2f', ...
      numel(ssi_before), mean(ssi_before)), 'FontSize', 11);
ylim(ax1, [-0.05 1.15]);
grid(ax1, 'on');  ax1.GridAlpha = 0.20;
text(ax1, ideal_power_target, 1.08, 'Ideal Subspace Alignment', 'FontSize', 9, 'Color', 0.4*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── Panel 2: After GEDAI  (Signal vs Noise) ──────────────────────────────
ax2 = subplot(1, 2, 2);
hold(ax2, 'on');

h_noise = scatter(ax2, lpow_artifacts, ssi_artifacts, 38, col_noise, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.50);
h_sig   = scatter(ax2, lpow_after,     ssi_after,     38, col_sig, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.70);

% Ideal alignment horizon and dataset-specific target star
yline(ax2, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
h_star = scatter(ax2, ideal_power_target, 1, 250, col_star, 'p', 'filled', ...
                  'MarkerEdgeColor', 'k', 'LineWidth', 1.0);

% 95% confidence ellipses
draw_ellipse(ax2, lpow_after,     ssi_after,     col_sig,   0.95);
draw_ellipse(ax2, lpow_artifacts, ssi_artifacts, col_noise, 0.95);

% ── 5.1 LDA Posterior Background (Creative Contour) ───────────────────────
if ~isnan(lda_accuracy) && exist('lda_mdl', 'var')
    % Create meshgrid for current limits
    xl = xlim(ax2); yl = ylim(ax2);
    nx = 100; ny = 100;
    [XG, YG] = meshgrid(linspace(xl(1), xl(2), nx), linspace(yl(1), yl(2), ny));
    
    % Prepare for prediction (must match model features: [SSI, Power])
    % Note: model was trained on [SSI, Power], so we use [YG(:), XG(:)] 
    % Wait, X_lda was [ssi_after, lpow_after] so feature 1=SSI, 2=Power.
    % In plot: X=Power, Y=SSI. So prediction input is [YG(:), XG(:)].
    try
        % Extract the trained discriminant model from CrossVal (first fold)
        mdl_for_bg = lda_mdl.Trained{1}; 
        [~, Post] = predict(mdl_for_bg, [YG(:), XG(:)]);
        
        % Probability of class 1 (Signal)
        ProbSig = reshape(Post(:, 2), ny, nx);
        
        % Create custom diverging colormap (Red -> Grey -> Green)
        cpoints = [0.85 0.13 0.13; 0.98 0.98 0.98; 0.08 0.72 0.22];
        interp_cmap = [linspace(cpoints(1,1), cpoints(2,1), 64)', linspace(cpoints(1,2), cpoints(2,2), 64)', linspace(cpoints(1,3), cpoints(2,3), 64)'; ...
                       linspace(cpoints(2,1), cpoints(3,1), 64)', linspace(cpoints(2,2), cpoints(3,2), 64)', linspace(cpoints(2,3), cpoints(3,3), 64)'];

        % Draw filled contours
        [~, h_cont] = contourf(ax2, XG, YG, ProbSig, linspace(0, 1, 15), 'LineColor', 'none', 'FaceAlpha', 0.15);
        colormap(ax2, interp_cmap);
        uistack(h_cont, 'bottom');
    catch
    end
end

ssi_diff = (mean(ssi_after) - mean(ssi_artifacts)) * 100;
if ~isnan(lda_accuracy)
    ttl = sprintf('After Denoising  |  LDA: %.1f%%  |  SSSI-NSSI: %.1f%%\nMean SSSI: %.2f   |   Mean NSSI: %.2f', ...
                  lda_accuracy, ssi_diff, mean(ssi_after), mean(ssi_artifacts));
else
    ttl = sprintf('After Denoising  |  SSSI-NSSI: %.1f%%\nMean SSSI: %.2f   |   Mean NSSI: %.2f', ...
                  ssi_diff, mean(ssi_after), mean(ssi_artifacts));
end
title(ax2, ttl, 'FontSize', 11);

xlabel(ax2, 'Epoch Power (dB)',                'FontSize', 11);
ylabel(ax2, sprintf('SSI  (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
legend(ax2, [h_star, h_sig, h_noise], ...
       {'Target Subspace', ...
        sprintf('Signal  (mean SSI=%.2f)', mean(ssi_after)), ...
        sprintf('Noise   (mean SSI=%.2f)', mean(ssi_artifacts))}, ...
       'Location', 'best', 'FontSize', 9);
ylim(ax2, [-0.05 1.15]);
grid(ax2, 'on');  ax2.GridAlpha = 0.20;
text(ax2, ideal_power_target, 1.08, 'Target Subspace', 'FontSize', 9, 'Color', 0.4*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── Match X-axis range (power): must include the full width of all 95% ellipses ──
% The horizontal extents of a 2D confidence ellipse are at MeanX +/- sqrt(VarX * Chi2)
chi2_95 = -2 * log(1 - 0.95);
get_extents = @(x) [mean(x) - sqrt(var(x)*chi2_95), mean(x) + sqrt(var(x)*chi2_95)];

ext_b = get_extents(lpow_before);
ext_a = get_extents(lpow_after);
ext_n = get_extents(lpow_artifacts);

% Union of all points and all ellipse extents
all_vals = [lpow_before; lpow_after; lpow_artifacts; ext_b'; ext_a'; ext_n'];
x_min = min(all_vals);
x_max = max(all_vals);
x_pad = 0.10 * (x_max - x_min);

x_lims = [x_min - x_pad, x_max + x_pad];
xlim(ax1, x_lims);
xlim(ax2, x_lims);

% ── Shared super-title ────────────────────────────────────────────────────
sgtitle('SENSAI visualization:  Subspace Similarity  vs  Epoch Power', ...
        'FontSize', 13, 'FontWeight', 'bold');

% ── 6. Output Metrics ─────────────────────────────────────────────────────
metrics = struct();
metrics.ssi_before_mean = mean(ssi_before);
metrics.ssi_after_mean  = mean(ssi_after);
metrics.ssi_artifacts_mean = mean(ssi_artifacts);
metrics.ssi_nssi_diff   = (mean(ssi_after) - mean(ssi_artifacts)) * 100;
metrics.lda_accuracy    = lda_accuracy;
metrics.ideal_power_target_db = ideal_power_target;

end


%% ══════════════════════════════════════════════════════════════════════════
function angs = extract_angles(C_array, basis_ref, top_PCs)
% Returns matrix (num_epochs × top_PCs) of cosines of principal angles
    n    = length(C_array);
    angs = zeros(n, top_PCs);
    for i = 1:n
        [V, D]    = eig(C_array{i});
        [~, idx]  = sort(diag(D), 'descend');
        basis_c   = V(:, idx(1:top_PCs));
        angs(i,:) = subspace_angles(basis_c, basis_ref)';
    end
end


%% ══════════════════════════════════════════════════════════════════════════
function pow = extract_power(C_array)
% Epoch power = trace(C) = total channel variance
    n   = length(C_array);
    pow = zeros(n, 1);
    for i = 1:n
        pow(i) = trace(C_array{i});
    end
end


%% ══════════════════════════════════════════════════════════════════════════
function draw_ellipse(ax, x, y, col, conf)
% Draw a confidence ellipse at level 'conf' (e.g. 0.95) for 2D data (x,y).
% Uses eigendecomposition of the sample covariance.
    if numel(x) < 3
        return;
    end
    data = [x(:), y(:)];
    mu   = mean(data, 1);
    C    = cov(data);

    % Chi-squared threshold for the requested confidence level
    chi2_val = -2 * log(1 - conf);      % equivalent to chi2inv(conf, 2)

    [V, D] = eig(C);
    radii  = sqrt(diag(D) * chi2_val);  % semi-axes

    % Parametric ellipse
    t  = linspace(0, 2*pi, 200);
    xy = V * (radii .* [cos(t); sin(t)]);   % rotate
    ex = mu(1) + xy(1,:);
    ey = mu(2) + xy(2,:);

    plot(ax, ex, ey, '-', 'Color', [col, 0.85], 'LineWidth', 1.8);
    plot(ax, mu(1), mu(2), '+', 'Color', col*0.7, ...
         'MarkerSize', 10, 'LineWidth', 2);
end