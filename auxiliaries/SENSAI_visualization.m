function metrics = SENSAI_visualization(EEGavRef, EEGclean, EEGartifacts, ref_cov, epoch_duration_sec, signal_type, SSI_top_PCs, artifact_threshold_type, smoothing_window_seconds, SENSAI_score, mean_ENOVA, epoch_size_in_cycles, lowcut_frequency)
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

%% ── 0. Parse inputs and parameters ──────────────────────────────────────
if nargin < 5 || isempty(epoch_duration_sec)
    epoch_duration_sec = 1; 
end
if nargin < 6 || isempty(signal_type)
    signal_type = 'eeg';
end
if nargin < 7 || isempty(SSI_top_PCs)
    if strcmpi(signal_type, 'meg') || size(ref_cov, 1) <= 100
        SSI_top_PCs = 4; 
    else
        SSI_top_PCs = 3; 
    end
end

%% ── 1. Epoch Data & Extract Covariances ─────────────────────────────────
srate = EEGavRef.srate;
epoch_samples = round(srate * epoch_duration_sec);

function COV_array = compute_cov_3d(data, epoch_samples)
    pnts = size(data, 2);
    remainder = rem(pnts, epoch_samples);
    if remainder ~= 0
        samples_to_pad = epoch_samples - remainder;
        padding = fliplr(data(:, end-samples_to_pad+1:end)); 
        data = [data, padding];
    end
    num_epochs = size(data, 2) / epoch_samples;
    n_chans = size(data, 1);
    try
        % Fast page-wise multiplication (introduced in R2020b)
        data_3d = reshape(data, n_chans, epoch_samples, num_epochs);
        data_3d = data_3d - mean(data_3d, 2);
        COV_array = pagemtimes(data_3d, 'none', data_3d, 'transpose') / (epoch_samples - 1);
    catch
        % Fallback for MATLAB versions prior to R2020b
        COV_array = zeros(n_chans, n_chans, num_epochs);
        for epo = 1:num_epochs
            i_start = (epo - 1) * epoch_samples + 1;
            i_end   = i_start + epoch_samples - 1;
            COV_array(:, :, epo) = cov(data(:, i_start:i_end)');
        end
    end
end

C_before    = compute_cov_3d(EEGavRef.data, epoch_samples);
C_after     = compute_cov_3d(EEGclean.data, epoch_samples);
C_artifacts = compute_cov_3d(EEGartifacts.data, epoch_samples);

num_epochs_after = size(C_after, 3);
num_epochs_before = size(C_before, 3);
if num_epochs_before > num_epochs_after
    C_before = C_before(:, :, 1:num_epochs_after);
elseif num_epochs_before < num_epochs_after
    C_after = C_after(:, :, 1:num_epochs_before);
    C_artifacts = C_artifacts(:, :, 1:num_epochs_before);
end

%% ── 2. Subspace Analysis ───────────────────────────────────────────────
[Vref, Dref] = eig(ref_cov);
[~, idx]     = sort(diag(Dref), 'descend');
basis_ref    = Vref(:, idx(1:SSI_top_PCs));

angs_before    = extract_angles(C_before,    basis_ref, SSI_top_PCs);
angs_after     = extract_angles(C_after,     basis_ref, SSI_top_PCs);
angs_artifacts = extract_angles(C_artifacts, basis_ref, SSI_top_PCs);

ssi_before    = prod(angs_before,    2) .^ (1/SSI_top_PCs);
ssi_after     = prod(angs_after,     2) .^ (1/SSI_top_PCs);
ssi_artifacts = prod(angs_artifacts, 2) .^ (1/SSI_top_PCs);

lpow_before    = 10 * log10(extract_power(C_before));
lpow_after     = 10 * log10(extract_power(C_after));
lpow_artifacts = 10 * log10(extract_power(C_artifacts));

ideal_power_target = median(lpow_after);

%% ── 3. LDA Classification ──────────────────────────────────────────────
X_lda = [ssi_after, lpow_after; ssi_artifacts, lpow_artifacts];
Y_lda = [ones(numel(ssi_after), 1); zeros(numel(ssi_artifacts), 1)];
try
    lda_mdl      = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5);
    lda_accuracy = (1 - kfoldLoss(lda_mdl)) * 100;
    lda_full     = fitcdiscr(X_lda, Y_lda);
    
    % --- SSI Silhouette Score ---
    % User preference: Only sensitive to the Y-axis (SSI) separation.
    sil_scores   = silhouette(X_lda(:, 1), Y_lda, 'sqEuclidean');
    sil_signal   = mean(sil_scores(Y_lda == 1));
catch
    lda_accuracy = NaN;
    lda_full     = [];
    sil_signal   = NaN;
end

%% ── 4. Plotting ─────────────────────────────────────────────────────────
if nargin >= 10 && ~isempty(SENSAI_score)
    if nargin < 11 || isempty(mean_ENOVA), mean_ENOVA = 0; end
    if nargin < 12 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
    if nargin < 13 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
    if isempty(artifact_threshold_type), artifact_threshold_type = 'auto'; end
    if isempty(smoothing_window_seconds), smoothing_window_seconds = Inf; end

    plot_title = ['SENSAI = ' num2str(round(SENSAI_score, 2, 'significant')) '%, ENOVA = ' num2str(round(mean_ENOVA * 100, 2, 'significant')) '% [' artifact_threshold_type ', ' num2str(epoch_size_in_cycles) ' cycles, ' num2str(lowcut_frequency) ' Hz, ' num2str(smoothing_window_seconds) ' s]'];
else
    plot_title = 'SENSAI visualization:  Subspace Similarity  vs  Epoch Power';
end

fig = figure('Name', plot_title, 'Color', 'w', 'Position', [80 100 1350 580], 'Visible', 'off');

col_sig  = [0.08 0.72 0.22];
col_noise= [0.85 0.13 0.13];
col_bef  = [0.30 0.45 0.75];
col_star = [1.00 0.88 0.00];

% --- Panel 1: Before GEDAI ---
figure(fig);
ax1 = subplot(1, 2, 1);
set(ax1, 'Position', [0.06, 0.12, 0.35, 0.74]); 
hold(ax1, 'on');

[~, si] = sort(ssi_before, 'ascend');
scatter(ax1, lpow_before(si), ssi_before(si), 38, ssi_before(si), ...
        'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);

yline(ax1, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
colormap(ax1, parula);
cb = colorbar(ax1, 'eastoutside');
cb.Label.String = 'SSI (Subspace Similarity Index) relative to Leadfield';
cb.Label.FontSize = 12;
ax1.CLim = [0 1];

xlabel(ax1, 'Epoch Power (dB)', 'FontSize', 11);
ylabel(ax1, sprintf('SSI (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
ylim(ax1, [-0.05 1.15]);
set(ax1, 'YTickMode', 'auto', 'YTickLabelMode', 'auto', 'TickDir', 'both');
grid(ax1, 'off');


% --- Panel 2: After GEDAI ---
figure(fig);
ax2 = subplot(1, 2, 2);
set(ax2, 'Position', [0.55, 0.12, 0.35, 0.74]); 
hold(ax2, 'on');

h_noise = scatter(ax2, lpow_artifacts, ssi_artifacts, 38, col_noise, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.40);
h_sig   = scatter(ax2, lpow_after, ssi_after, 38, col_sig, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.40);

yline(ax2, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
h_star = scatter(ax2, ideal_power_target, 1, 250, col_star, 'p', 'filled', 'MarkerEdgeColor', 'k');

ylim(ax2, [-0.05 1.15]);
ylabel(ax2, sprintf('SSI (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
set(ax2, 'YTickMode', 'auto', 'YTickLabelMode', 'auto', 'TickDir', 'both');
grid(ax2, 'off');

% ── 5.0 Match X-axis range (power) ──
chi2_95 = -2 * log(1 - 0.95);
get_extents = @(x) [mean(x) - sqrt(var(x)*chi2_95), mean(x) + sqrt(var(x)*chi2_95)];
ext_b = get_extents(lpow_before); ext_a = get_extents(lpow_after); ext_n = get_extents(lpow_artifacts);
all_vals = [lpow_before; lpow_after; lpow_artifacts; ext_b'; ext_a'; ext_n'];
all_vals_finite = all_vals(isfinite(all_vals));
if isempty(all_vals_finite)
    x_lims = [-10, 10];
else
    x_min = min(all_vals_finite);
    x_max = max(all_vals_finite);
    if x_min == x_max
        x_lims = [x_min - 5, x_max + 5];
    else
        x_lims = [x_min - 2, x_max + 5];
    end
end
xlim(ax1, x_lims); xlim(ax2, x_lims);
text(ax1, mean(x_lims), 1.10, 'Leadfield Subspace', 'FontSize', 10, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── 5.1 LDA Shading ──
if ~isempty(lda_full)
    xl = ax2.XLim; yl = ax2.YLim;
    [Xg, Yg] = meshgrid(linspace(xl(1), xl(2), 200), linspace(yl(1), yl(2), 200));
    try
        coeff = lda_full.Coeffs(1, 2);
        scores = [Yg(:), Xg(:)] * coeff.Linear + coeff.Const;
        Pg = reshape(1 ./ (1 + exp(scores)), size(Xg));
    catch
        [~, Probs] = predict(lda_full, [Yg(:), Xg(:)]);
        Pg = reshape(Probs(:,2), size(Xg));
    end 
    h_cont = imagesc(ax2, xl, yl, Pg);
    set(ax2, 'YDir', 'normal'); 
    n = 64;
    bg_cmap = [[ones(n,1); linspace(1, 0.92, n)'], [linspace(0.92, 1, n)'; ones(n,1)], [linspace(0.92, 1, n)'; linspace(1, 0.92, n)']];
    colormap(ax2, bg_cmap); ax2.CLim = [0 1];
    uistack(h_cont, 'bottom');
end
text(ax2, ideal_power_target, 1.10, 'Leadfield Subspace', 'FontSize', 10, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

xlabel(ax2, 'Epoch Power (dB)', 'FontSize', 11);
legend(ax2, [h_star, h_sig, h_noise], {'Leadfield Subspace', sprintf('Signal (mean SSI=%.2f)', mean(ssi_after)), sprintf('Noise (mean SSI=%.2f)', mean(ssi_artifacts))}, ...
       'Location', 'northeastoutside', 'FontSize', 10);

try
    sgtitle(plot_title, 'FontSize', 13, 'FontWeight', 'bold');
catch
    try
        suptitle(plot_title);
    catch
    end
end

% ── 6. Add Marginal Density Distributions ─────────────────────────────────
% Panel 1 Marginals (Empty data, just for layout/title alignment)
ttl1 = sprintf('Before Denoising  |  Mean SSI: %.2f', mean(ssi_before));
add_marginal_densities(ax1, {}, {}, {}, SSI_top_PCs, ttl1);

% Panel 2 Marginals
if ~isnan(sil_signal)
    ttl2 = sprintf('After Denoising  |  Mean SSSI: %.2f   |   Mean NSSI: %.2f\nSSI Silhouette Score: %.2f', ...
                  mean(ssi_after), mean(ssi_artifacts), sil_signal);
else
    ttl2 = sprintf('After Denoising\nMean SSSI: %.2f   |   Mean NSSI: %.2f', ...
                  mean(ssi_after), mean(ssi_artifacts));
end
add_marginal_densities(ax2, {lpow_after, lpow_artifacts}, {ssi_after, ssi_artifacts}, {col_sig, col_noise}, SSI_top_PCs, ttl2);

metrics = struct('ssi_before_mean', mean(ssi_before), ...
                 'ssi_after_mean', mean(ssi_after), ...
                 'ssi_artifacts_mean', mean(ssi_artifacts), ...
                 'lda_accuracy', lda_accuracy, ...
                 'signal_silhouette', sil_signal, ...
                 'ideal_power_target_db', ideal_power_target);

% --- Instant Reveal ---
set(fig, 'Visible', 'on');
drawnow;

end

%% Helper Functions
function angs = extract_angles(C, basis_ref, top_PCs)
    n = size(C, 3);
    angs = zeros(n, top_PCs);
    try
        % Faster 'vector' option (introduced in R2016b)
        for i = 1:n
            [V, d] = eig(C(:, :, i), 'vector');
            [~, idx] = sort(d, 'descend');
            basis_c = V(:, idx(1:top_PCs));
            angs(i, :) = svd(double(basis_c)' * double(basis_ref))';
        end
    catch
        % Fallback for MATLAB versions prior to R2016b
        for i = 1:n
            [V, D] = eig(C(:, :, i));
            [~, idx] = sort(diag(D), 'descend');
            basis_c = V(:, idx(1:top_PCs));
            angs(i, :) = svd(double(basis_c)' * double(basis_ref))';
        end
    end
end

function pow = extract_power(C)
    n_chans = size(C, 1);
    n_epochs = size(C, 3);
    C_reshaped = reshape(C, [], n_epochs);
    diag_indices = 1 : (n_chans + 1) : (n_chans^2);
    pow = sum(C_reshaped(diag_indices, :), 1)';
end

function draw_ellipse(ax, x, y, col, conf)
    if numel(x) < 3, return; end
    mu = mean([x(:), y(:)], 1); C = cov([x(:), y(:)]); chi2_val = -2 * log(1 - conf);
    [V, D] = eig(C); radii = sqrt(diag(D) * chi2_val);
    t = linspace(0, 2*pi, 200); xy = V * (radii .* [cos(t); sin(t)]);
    plot(ax, mu(1) + xy(1,:), mu(2) + xy(2,:), '-', 'Color', [col, 0.85], 'LineWidth', 1.8);
    plot(ax, mu(1), mu(2), '+', 'Color', col*0.7, 'MarkerSize', 10, 'LineWidth', 2);
end

function add_marginal_densities(ax, x_data, y_data, cols, SSI_top_PCs, ttl)
    set(ax, 'Units', 'normalized'); 
    pos = ax.Position; 
    margin = 0.01; 
    m_size = 0.12;
    
    % Strict manual sizing to ensure perfect symmetry between left and right panels
    new_w = 0.28; 
    new_h = 0.65;
    
    set(ax, 'Position', [pos(1), pos(2), new_w, new_h]);
    ax_x = axes('Position', [pos(1), pos(2) + new_h + margin, new_w, pos(4) * m_size], 'Units', 'normalized', 'Color', 'none', 'XLim', ax.XLim, 'XTick', [], 'YTick', [], 'XAxisLocation', 'bottom', 'Box', 'off');
    hold(ax_x, 'on'); title(ax_x, ttl, 'FontSize', 11);
    ax_y = axes('Position', [pos(1) + new_w + margin, pos(2), pos(3) * m_size, new_h], 'Units', 'normalized', 'Color', 'none', 'YLim', ax.YLim, 'YTick', [], 'XTick', [], 'YAxisLocation', 'left', 'Box', 'off');
    hold(ax_y, 'on');
    for i = 1:length(x_data)
        try [f, xi] = ksdensity(x_data{i}); fill(ax_x, xi, f, cols{i}, 'FaceAlpha', 0.2, 'EdgeColor', cols{i}, 'LineWidth', 1); catch, end
        try [f, yi] = ksdensity(y_data{i}); fill(ax_y, f, yi, cols{i}, 'FaceAlpha', 0.2, 'EdgeColor', cols{i}, 'LineWidth', 1); catch, end
    end
    linkaxes([ax, ax_x], 'x'); linkaxes([ax, ax_y], 'y');
    if isempty(x_data) && isempty(y_data)
        ax_x.XAxis.Visible = 'off'; ax_x.YAxis.Visible = 'off';
        ax_y.XAxis.Visible = 'off'; ax_y.YAxis.Visible = 'off';
    end
    uistack(ax, 'top');
end