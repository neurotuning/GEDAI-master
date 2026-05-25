function metrics = SENSAI_visualization(EEGavRef, EEGclean, EEGartifacts, ref_cov, epoch_duration_sec, signal_type, SSI_top_PCs, artifact_threshold_type, smoothing_window_seconds, SENSAI_score, mean_ENOVA)
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
ref_cov = real(ref_cov);
ref_cov = (ref_cov + ref_cov') / 2;

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

function COV_array = compute_cov_array(data)
    pnts = size(data, 2);
    remainder = rem(pnts, epoch_samples);
    if remainder ~= 0
        samples_to_pad = epoch_samples - remainder;
        padding = fliplr(data(:, end-samples_to_pad+1:end)); 
        data = [data, padding];
    end
    num_epochs = size(data, 2) / epoch_samples;
    num_chans = size(data, 1);
    
    data3D = reshape(data, num_chans, epoch_samples, num_epochs);
    data3D = data3D - mean(data3D, 2); % Center data for covariance
    
    if exist('pagemtimes', 'builtin')
        COV_array = pagemtimes(data3D, permute(data3D, [2 1 3])) / (epoch_samples - 1);
    else
        COV_array = zeros(num_chans, num_chans, num_epochs);
        for epo = 1:num_epochs
            COV_array(:,:,epo) = (data3D(:,:,epo) * data3D(:,:,epo)') / (epoch_samples - 1);
        end
    end
    % Enforce real and symmetric covariance matrices
    COV_array = real(COV_array);
    for epo = 1:num_epochs
        COV_array(:,:,epo) = (COV_array(:,:,epo) + COV_array(:,:,epo)') / 2;
    end
end

C_before    = compute_cov_array(EEGavRef.data);
C_after     = compute_cov_array(EEGclean.data);
C_artifacts = compute_cov_array(EEGartifacts.data);

num_epochs_after = size(C_after, 3);
if size(C_before, 3) > num_epochs_after
    C_before = C_before(:, :, 1:num_epochs_after);
elseif size(C_before, 3) < num_epochs_after
    C_after = C_after(:, :, 1:size(C_before, 3));
    C_artifacts = C_artifacts(:, :, 1:size(C_before, 3));
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

%% ── 3. GMM Clustering ──────────────────────────────────────────────────
X_data = [ssi_after, lpow_after; ssi_artifacts, lpow_artifacts];
Y_true = [ones(numel(ssi_after), 1); zeros(numel(ssi_artifacts), 1)];
try
    % Fit a 2-component GMM with unshared, full covariance matrices
    gmm = fitgmdist(X_data, 2, 'Replicates', 3, 'RegularizationValue', 1e-5);
    
    % Map GMM components to Signal (1) and Noise (0)
    Probs_points = posterior(gmm, X_data);
    % Identify which GMM component corresponds to the "Signal" class (majority of ssi_after points)
    [~, signal_comp] = max([mean(Probs_points(1:numel(ssi_after), 1)), mean(Probs_points(1:numel(ssi_after), 2))]);
    noise_comp = 3 - signal_comp;
    
    % Compute unsupervised GMM clustering accuracy against actual partition
    gmm_labels = cluster(gmm, X_data);
    predicted_labels = (gmm_labels == signal_comp);
    gmm_accuracy = mean(predicted_labels == Y_true) * 100;
    
    % Compute Adjusted Rand Index (ARI)
    gmm_ari = compute_ari(Y_true, predicted_labels);
    
    % --- SSI Silhouette Score ---
    % Only sensitive to the Y-axis (SSI) separation.
    sil_signal = custom_1d_silhouette(X_data(:, 1), Y_true, 1);
catch ME
    warning('GMM fitting failed: %s', ME.message);
    disp(ME.getReport());
    gmm = [];
    gmm_accuracy = NaN;
    gmm_ari = NaN;
    signal_comp = 1;
    sil_signal   = NaN;
end

%% ── 4. Plotting ─────────────────────────────────────────────────────────
if nargin >= 10 && ~isempty(artifact_threshold_type) && ~isempty(smoothing_window_seconds) && ~isempty(SENSAI_score)
    if nargin >= 11 && ~isempty(mean_ENOVA)
        plot_title = ['SENSAI visualization (' artifact_threshold_type ' | Window: ' num2str(smoothing_window_seconds) ' s | SENSAI: ' num2str(round(SENSAI_score, 1)) '% | ENOVA: ' num2str(round(mean_ENOVA*100, 1)) '%)'];
    else
        plot_title = ['SENSAI visualization (' artifact_threshold_type ' | Window: ' num2str(smoothing_window_seconds) ' s | SENSAI: ' num2str(round(SENSAI_score, 1)) '%)'];
    end
else
    plot_title = 'SENSAI visualization:  Subspace Similarity  vs  Epoch Power';
end

fig = figure('Name', plot_title, 'Color', 'w', 'Position', [80 100 1350 580], 'Visible', 'off');

col_sig  = [0.08 0.72 0.22];
col_noise= [0.85 0.13 0.13];
col_bef  = [0.30 0.45 0.75];
col_star = [1.00 0.88 0.00];

% --- Panel 1: Before GEDAI ---
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
clim(ax1, [0 1]);

xlabel(ax1, 'Epoch Power (dB)', 'FontSize', 11);
ylabel(ax1, sprintf('SSI (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
ylim(ax1, [-0.05 1.15]);
set(ax1, 'YTickMode', 'auto', 'YTickLabelMode', 'auto', 'TickDir', 'both');
grid(ax1, 'off');


% --- Panel 2: After GEDAI ---
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
x_min = min(all_vals); x_max = max(all_vals);
x_lims = [x_min - 2, x_max + 5]; 
xlim(ax1, x_lims); xlim(ax2, x_lims);
text(ax1, mean(x_lims), 1.10, 'Leadfield Subspace', 'FontSize', 10, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── 5.1 GMM Clustering Shading ──
if ~isempty(gmm)
    xl = ax2.XLim; yl = ax2.YLim;
    [Xg, Yg] = meshgrid(linspace(xl(1), xl(2), 200), linspace(yl(1), yl(2), 200));
    Probs = posterior(gmm, [Yg(:), Xg(:)]);
    Pg = reshape(Probs(:, signal_comp), size(Xg)); 
    h_cont = imagesc(ax2, xl, yl, Pg);
    set(ax2, 'YDir', 'normal'); 
    n = 64;
    bg_cmap = [[ones(n,1); linspace(1, 0.92, n)'], [linspace(0.92, 1, n)'; ones(n,1)], [linspace(0.92, 1, n)'; linspace(1, 0.92, n)']];
    colormap(ax2, bg_cmap); clim(ax2, [0 1]);
    uistack(h_cont, 'bottom');
end
text(ax2, ideal_power_target, 1.10, 'Leadfield Subspace', 'FontSize', 10, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

xlabel(ax2, 'Epoch Power (dB)', 'FontSize', 11);
legend(ax2, [h_star, h_sig, h_noise], {'Leadfield Subspace', sprintf('Signal (mean SSI=%.2f)', mean(ssi_after)), sprintf('Noise (mean SSI=%.2f)', mean(ssi_artifacts))}, ...
       'Location', 'northeastoutside', 'FontSize', 10);

sgtitle(plot_title, 'FontSize', 13, 'FontWeight', 'bold');

% ── 6. Add Marginal Density Distributions ─────────────────────────────────
% Panel 1 Marginals (Empty data, just for layout/title alignment)
ttl1 = sprintf('Before Denoising  |  Mean SSI: %.2f', mean(ssi_before));
add_marginal_densities(ax1, {}, {}, {}, SSI_top_PCs, ttl1);

% Panel 2 Marginals
if ~isnan(gmm_ari)
    ttl2 = sprintf('After Denoising  |  Mean SSSI: %.2f   |   Mean NSSI: %.2f\nGMM Adjusted Rand Index (GARI): %.3f', ...
                  mean(ssi_after), mean(ssi_artifacts), gmm_ari);
else
    ttl2 = sprintf('After Denoising\nMean SSSI: %.2f   |   Mean NSSI: %.2f', ...
                  mean(ssi_after), mean(ssi_artifacts));
end
add_marginal_densities(ax2, {lpow_after, lpow_artifacts}, {ssi_after, ssi_artifacts}, {col_sig, col_noise}, SSI_top_PCs, ttl2);

metrics = struct('ssi_before_mean', mean(ssi_before), ...
                 'ssi_after_mean', mean(ssi_after), ...
                 'ssi_artifacts_mean', mean(ssi_artifacts), ...
                 'gmm_accuracy', gmm_accuracy, ...
                 'lda_accuracy', gmm_accuracy, ... % Included for backward compatibility
                 'GARI', gmm_ari, ...
                 'signal_silhouette', sil_signal, ...
                 'ideal_power_target_db', ideal_power_target);
if nargin >= 11 && ~isempty(mean_ENOVA)
    metrics.mean_enova = mean_ENOVA;
end

% --- Instant Reveal ---
set(fig, 'Visible', 'on');
drawnow;

end

%% Helper Functions
function angs = extract_angles(C_array, basis_ref, top_PCs)
    n = size(C_array, 3); angs = zeros(n, top_PCs);
    for i = 1:n
        C_array(:,:,i) = (C_array(:,:,i) + C_array(:,:,i)') / 2;
        [V, D] = eig(C_array(:,:,i)); [~, idx] = sort(diag(D), 'descend');
        basis_c = V(:, idx(1:top_PCs));
        angs(i,:) = subspace_angles(basis_c, basis_ref)';
    end
end

function pow = extract_power(C_array)
    n = size(C_array, 3); pow = zeros(n, 1);
    for i = 1:n, pow(i) = trace(C_array(:,:,i)); end
end

function draw_ellipse(ax, x, y, col, conf)
    if numel(x) < 3, return; end
    mu = mean([x(:), y(:)], 1); C = cov([x(:), y(:)]); chi2_val = -2 * log(1 - conf);
    C = (C + C') / 2;
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

function sil_score = custom_1d_silhouette(x, y, target_class)
    % Custom 1D silhouette calculation using sqEuclidean distance
    idx_target = find(y == target_class);
    idx_other = find(y ~= target_class);
    n_target = length(idx_target);
    n_other = length(idx_other);
    
    if n_target <= 1 || n_other == 0
        sil_score = NaN;
        return;
    end
    
    x_target = x(idx_target);
    x_other = x(idx_other);
    sil_scores = zeros(n_target, 1);
    
    for i = 1:n_target
        a_i = sum((x_target(i) - x_target).^2) / (n_target - 1);
        b_i = sum((x_target(i) - x_other).^2) / n_other;
        if max(a_i, b_i) == 0
            sil_scores(i) = 0;
        else
            sil_scores(i) = (b_i - a_i) / max(a_i, b_i);
        end
    end
    sil_score = mean(sil_scores);
end

function ari = compute_ari(labels_true, labels_pred)
    % Manual 2x2 contingency table for binary classes {0, 1}
    C = zeros(2, 2);
    for i = 1:numel(labels_true)
        row = double(labels_true(i)) + 1;
        col = double(labels_pred(i)) + 1;
        C(row, col) = C(row, col) + 1;
    end
    
    n = sum(C(:));
    sum_rows = sum(C, 2);
    sum_cols = sum(C, 1);
    
    % Sum of combinations
    n_choose_2 = n * (n - 1) / 2;
    sum_nij_choose_2 = sum(C(:) .* (C(:) - 1)) / 2;
    sum_ai_choose_2 = sum(sum_rows .* (sum_rows - 1)) / 2;
    sum_bj_choose_2 = sum(sum_cols .* (sum_cols - 1)) / 2;
    
    % Expected Index
    expected_index = sum_ai_choose_2 * sum_bj_choose_2 / n_choose_2;
    max_index = (sum_ai_choose_2 + sum_bj_choose_2) / 2;
    
    % Adjusted Rand Index
    if max_index == expected_index
        ari = 0;
    else
        ari = (sum_nij_choose_2 - expected_index) / (max_index - expected_index);
    end
end