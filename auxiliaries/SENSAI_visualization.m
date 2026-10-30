function metrics = SENSAI_visualization(EEGavRef, EEGclean, EEGartifacts, ref_cov, epoch_duration_sec, signal_type, SSI_top_PCs, artifact_threshold_type, smoothing_window_seconds, SENSAI_score, mean_ENOVA, epoch_size_in_cycles, lowcut_frequency, ENOVA_per_channel)
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
    SSI_top_PCs = 3; 
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
if ndims(ref_cov) == 3
    ref_cov = mean(ref_cov, 3);
end
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
    if isnumeric(artifact_threshold_type)
        artifact_threshold_str = num2str(artifact_threshold_type, '%.2f');
    elseif isstring(artifact_threshold_type)
        artifact_threshold_str = char(artifact_threshold_type);
    else
        artifact_threshold_str = artifact_threshold_type;
    end
    if isempty(smoothing_window_seconds), smoothing_window_seconds = Inf; end

    spacer = '              '; % Doubled whitespace separation (2 tabs)
    plot_title = ['SENSAI = ' num2str(round(SENSAI_score, 2, 'significant')) '%' spacer 'ENOVA = ' num2str(round(mean_ENOVA * 100, 2, 'significant')) '%' spacer '[' artifact_threshold_str ', ' num2str(epoch_size_in_cycles) ' cycles, ' num2str(lowcut_frequency) ' Hz, ' num2str(smoothing_window_seconds) ' s]'];
else
    plot_title = 'SENSAI visualization:  Subspace Similarity  vs  Epoch Power';
end

fig = figure('Name', plot_title, 'Color', 'w', 'Position', [40 80 1700 580], 'Visible', 'off');

col_sig  = [0.08 0.72 0.22];
col_noise= [0.85 0.13 0.13];
col_bef  = [0.30 0.45 0.75];
col_star = [1.00 0.88 0.00];

% --- Panel 1: Before GEDAI ---
set(groot, 'CurrentFigure', fig);
ax1 = subplot(1, 3, 1, 'Parent', fig);
set(ax1, 'Position', [0.045, 0.12, 0.22, 0.60], 'FontSize', 13); 
hold(ax1, 'on');

[~, si] = sort(ssi_before, 'ascend');
scatter(ax1, lpow_before(si), ssi_before(si), 38, ssi_before(si), ...
        'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);

yline(ax1, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
colormap(ax1, parula);
cb = colorbar(ax1, 'Position', [0.272, 0.12, 0.010, 0.60]);
cb.Label.String = 'SSI (Subspace Similarity Index) relative to Leadfield';
cb.Label.FontSize = 14; cb.FontSize = 12;
cb.Ticks = [];
ax1.CLim = [0 1];

xlabel(ax1, 'Epoch Power (dB)', 'FontSize', 14.5, 'FontWeight', 'bold');
ylabel(ax1, sprintf('SSI (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 14.5, 'FontWeight', 'bold');
ylim(ax1, [-0.05 1.15]);
set(ax1, 'YTickMode', 'auto', 'YTickLabelMode', 'auto', 'TickDir', 'both');
grid(ax1, 'off');


% --- Panel 2: After GEDAI ---
set(groot, 'CurrentFigure', fig);
ax2 = subplot(1, 3, 2, 'Parent', fig);
set(ax2, 'Position', [0.355, 0.12, 0.22, 0.60], 'FontSize', 13); 
hold(ax2, 'on');

h_noise = scatter(ax2, lpow_artifacts, ssi_artifacts, 38, col_noise, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.40);
h_sig   = scatter(ax2, lpow_after, ssi_after, 38, col_sig, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.40);

yline(ax2, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
h_star = scatter(ax2, ideal_power_target, 1, 250, col_star, 'p', 'filled', 'MarkerEdgeColor', 'k');

ylim(ax2, [-0.05 1.15]);
% Redundant y-axis label removed on Panel 2 as requested
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
text(ax1, mean(x_lims), 1.10, 'Leadfield Subspace', 'FontSize', 12.5, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

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
text(ax2, ideal_power_target, 1.10, 'Leadfield Subspace', 'FontSize', 12.5, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

xlabel(ax2, 'Epoch Power (dB)', 'FontSize', 14.5, 'FontWeight', 'bold');
legend(ax2, [h_star, h_sig, h_noise], {'Leadfield Subspace', sprintf('Signal (mean SSI=%.2f)', mean(ssi_after)), sprintf('Noise (mean SSI=%.2f)', mean(ssi_artifacts))}, ...
       'Position', [0.585, 0.77, 0.085, 0.08], 'FontSize', 11);

try
    sgtitle(fig, plot_title, 'FontSize', 16, 'FontWeight', 'bold');
catch
    try
        suptitle(plot_title);
    catch
    end
end

% ── 6. Add Marginal Density Distributions ─────────────────────────────────
% Panel 1 Marginals (Empty data, just for layout/title alignment)
ttl1 = sprintf('Before Denoising\nMean SSI: %.2f', mean(ssi_before));
add_marginal_densities(ax1, {}, {}, {}, SSI_top_PCs, ttl1);

% Panel 2 Marginals
if ~isnan(sil_signal)
    ttl2 = sprintf('After Denoising\nSSI Silhouette Score: %.2f', sil_signal);
else
    ttl2 = 'After Denoising';
end
add_marginal_densities(ax2, {lpow_after, lpow_artifacts}, {ssi_after, ssi_artifacts}, {col_sig, col_noise}, SSI_top_PCs, ttl2);

% --- Far-Right Panels: Dual Topoplots (Denoised Power [Upper] & ENOVA [Lower]) ---
set(groot, 'CurrentFigure', fig);

% 1. Compute Denoised Channel Power
pow_clean_raw = var(double(EEGclean.data), 0, 2);
total_gfp     = sum(pow_clean_raw);
pow_clean_gfp = (pow_clean_raw ./ max(total_gfp, eps)) * 100;
pow_clean_db  = 10 * log10(max(pow_clean_raw, eps));

% 2. Compute ENOVA per Channel
if nargin >= 14 && ~isempty(ENOVA_per_channel)
    enova_ch = ENOVA_per_channel;
elseif isfield(EEGclean, 'etc') && isfield(EEGclean.etc, 'GEDAI') && isfield(EEGclean.etc.GEDAI, 'ENOVA_per_channel') && ~isempty(EEGclean.etc.GEDAI.ENOVA_per_channel)
    enova_ch = EEGclean.etc.GEDAI.ENOVA_per_channel;
else
    var_noise = var(double(EEGartifacts.data), 0, 2);
    var_orig  = var(double(EEGavRef.data(:, 1:size(EEGartifacts.data, 2))), 0, 2);
    enova_ch  = var_noise ./ max(var_orig, eps);
end

enova_plot_val = double(enova_ch(:));
% Convert proportion (0-1) to percentage if not already in percent
finite_vals = enova_plot_val(isfinite(enova_plot_val));
if ~isempty(finite_vals) && max(finite_vals) <= 1.5
    enova_plot_val = enova_plot_val * 100;
end
% Clamp between 0% and 100% for topoplot (bad/flat channels get 100%)
enova_plot_val(isinf(enova_plot_val) | enova_plot_val > 100) = 100;
enova_plot_val(isnan(enova_plot_val)) = 0;

chanlocs = EEGclean.chanlocs;

% Ensure all channels have valid coordinates by auto-matching against standard 10-20 templates if needed
if isfield(chanlocs, 'labels') && ~isempty(chanlocs)
    needs_coord = false(1, length(chanlocs));
    for cIdx = 1:length(chanlocs)
        has_polar = isfield(chanlocs(cIdx), 'theta') && ~isempty(chanlocs(cIdx).theta) && ...
                    isnumeric(chanlocs(cIdx).theta) && ~isnan(chanlocs(cIdx).theta) && ...
                    isfield(chanlocs(cIdx), 'radius') && ~isempty(chanlocs(cIdx).radius) && ...
                    isnumeric(chanlocs(cIdx).radius) && ~isnan(chanlocs(cIdx).radius);
        if ~has_polar
            needs_coord(cIdx) = true;
        end
    end
    
    if any(needs_coord)
        ced_path = which('Standard-10-20-Cap81.ced');
        if isempty(ced_path)
            ced_path = which('Standard-10-20-Cap19.ced');
        end
        if ~isempty(ced_path) && exist('readlocs', 'file')
            try
                tmpl_locs = readlocs(ced_path);
                tmpl_labels = upper(strtrim({tmpl_locs.labels}));
                for cIdx = find(needs_coord)
                    curr_lbl = upper(strtrim(chanlocs(cIdx).labels));
                    match_pos = find(strcmp(curr_lbl, tmpl_labels), 1);
                    if ~isempty(match_pos)
                        chanlocs(cIdx).theta  = tmpl_locs(match_pos).theta;
                        chanlocs(cIdx).radius = tmpl_locs(match_pos).radius;
                        chanlocs(cIdx).X      = tmpl_locs(match_pos).X;
                        chanlocs(cIdx).Y      = tmpl_locs(match_pos).Y;
                        chanlocs(cIdx).Z      = tmpl_locs(match_pos).Z;
                        if isfield(tmpl_locs, 'sph_theta'), chanlocs(cIdx).sph_theta = tmpl_locs(match_pos).sph_theta; end
                        if isfield(tmpl_locs, 'sph_phi'),   chanlocs(cIdx).sph_phi   = tmpl_locs(match_pos).sph_phi;   end
                        if isfield(tmpl_locs, 'sph_radius'),chanlocs(cIdx).sph_radius= tmpl_locs(match_pos).sph_radius;end
                    end
                end
            catch
            end
        end
    end
end

% A topoplot requires at least 3 valid channels; otherwise fall back to bar plot
num_valid_coords = 0;
if isfield(chanlocs, 'theta')
    valid_mask = ~cellfun(@isempty, {chanlocs.theta}) & ~cellfun(@(x) any(isnan(x)), {chanlocs.theta});
    num_valid_coords = sum(valid_mask);
end
has_coords = (num_valid_coords >= 3);

% --- Upper Panel: Average PSD Across Electrodes (0.5 Hz Resolution) Before vs After ---
ax_psd = axes('Parent', fig, 'Position', [0.690, 0.510, 0.250, 0.210]);
hold(ax_psd, 'on');

% Compute PSD at exact 0.5 Hz resolution (2-second window)
srate_psd = EEGclean.srate;
nfft_psd = round(2 * srate_psd);
noverlap_psd = round(nfft_psd / 2);
w_psd = 0.5 * (1 - cos(2 * pi * (0:nfft_psd-1)' / (nfft_psd - 1)));

data_before = double(EEGavRef.data(:, 1:size(EEGclean.data, 2)));
data_after  = double(EEGclean.data);
data_noise  = double(EEGartifacts.data(:, 1:size(EEGclean.data, 2)));

if exist('pwelch', 'file')
    [pxx_b, f_psd] = pwelch(data_before', w_psd, noverlap_psd, nfft_psd, srate_psd);
    [pxx_a, ~]     = pwelch(data_after', w_psd, noverlap_psd, nfft_psd, srate_psd);
    [pxx_n, ~]     = pwelch(data_noise', w_psd, noverlap_psd, nfft_psd, srate_psd);
else
    [pxx_b, f_psd] = SENSAI_compute_welch_psd(data_before, srate_psd, nfft_psd, w_psd, noverlap_psd);
    [pxx_a, ~]     = SENSAI_compute_welch_psd(data_after, srate_psd, nfft_psd, w_psd, noverlap_psd);
    [pxx_n, ~]     = SENSAI_compute_welch_psd(data_noise, srate_psd, nfft_psd, w_psd, noverlap_psd);
end

% ENOVA-weighted PSD across electrodes
w_ch = enova_plot_val(:);
if sum(w_ch) <= 0 || any(isnan(w_ch))
    w_ch = ones(size(w_ch)) / length(w_ch);
else
    w_ch = w_ch / sum(w_ch);
end

psd_b_weighted = pxx_b * w_ch;
psd_a_weighted = pxx_a * w_ch;
psd_n_weighted = pxx_n * w_ch;

psd_b_db = 10 * log10(max(psd_b_weighted, eps));
psd_a_db = 10 * log10(max(psd_a_weighted, eps));
psd_n_db = 10 * log10(max(psd_n_weighted, eps));

% Restrict display to clinical EEG range (0.5 Hz to 45 Hz)
max_f_disp = min(45, srate_psd / 2);
f_mask = (f_psd >= 0.5 & f_psd <= max_f_disp);
f_disp = f_psd(f_mask);
pb_disp = psd_b_db(f_mask);
pa_disp = psd_a_db(f_mask);
pn_disp = psd_n_db(f_mask);

h_psd_b = plot(ax_psd, f_disp, pb_disp, '-', 'Color', [0.25 0.45 0.75], 'LineWidth', 1.8);
h_psd_a = plot(ax_psd, f_disp, pa_disp, '-', 'Color', [0.08 0.68 0.25], 'LineWidth', 2.0);
h_psd_n = plot(ax_psd, f_disp, pn_disp, '-', 'Color', [0.85 0.18 0.18], 'LineWidth', 1.8);

xlim(ax_psd, [0.5, max_f_disp]);
xlabel(ax_psd, 'Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax_psd, 'Power (dB)', 'FontSize', 14, 'FontWeight', 'bold');
title(ax_psd, 'ENOVA-weighted PSD over channels', 'FontSize', 14.5, 'FontWeight', 'bold');
grid(ax_psd, 'on');
box(ax_psd, 'off');
set(ax_psd, 'Color', 'w', 'FontSize', 13);
legend(ax_psd, [h_psd_b, h_psd_a, h_psd_n], {'Original', 'Signal', 'Noise'}, 'Location', 'northeast', 'FontSize', 11.5, 'Box', 'off');

% --- Lower Topoplot: Post-Denoising ENOVA per Channel ---
ax_topo_enova = axes('Parent', fig, 'Position', [0.720, 0.120, 0.14, 0.22]);
mean_enova_disp = mean(enova_plot_val, 'omitnan');

plot_done = false;
if has_coords && exist('topoplot', 'file')
    set(fig, 'CurrentAxes', ax_topo_enova);
    try
        topoplot(enova_plot_val, chanlocs, 'style', 'both', 'electrodes', 'on', ...
                 'shading', 'flat', 'gridscale', 120, 'conv', 'off', 'headrad', 'rim', 'whitebk', 'on');
        colormap(ax_topo_enova, 'turbo');
        cb_enova = colorbar(ax_topo_enova, 'Position', [0.875, 0.135, 0.009, 0.19]);
        cb_enova.Label.String = 'ENOVA (%)';
        cb_enova.Label.FontSize = 13.5; cb_enova.FontSize = 12;
        cb_enova.Label.FontWeight = 'bold';
        caxis(ax_topo_enova, [0, 100]);
        title(ax_topo_enova, 'ENOVA (Explained Noise Variance)', ...
              'FontSize', 14, 'FontWeight', 'bold');
        plot_done = true;
    catch ME_topo
        warning('SENSAI:topoplotFallback', 'Topoplot failed (%s). Falling back to bar plot.', ME_topo.message);
        cla(ax_topo_enova);
    end
end
if ~plot_done
    set(fig, 'CurrentAxes', ax_topo_enova);
    bar(ax_topo_enova, 1:length(enova_plot_val), enova_plot_val, 'FaceColor', [0.85 0.35 0.25], 'EdgeColor', 'none');
    ax_topo_enova.XTick = 1:length(enova_plot_val);
    if isfield(chanlocs, 'labels')
        ax_topo_enova.XTickLabel = {chanlocs.labels};
        ax_topo_enova.XTickLabelRotation = 45;
    end
    ylabel(ax_topo_enova, 'ENOVA (%)', 'FontSize', 14, 'FontWeight', 'bold');
    title(ax_topo_enova, 'ENOVA (Explained Noise Variance)', ...
          'FontSize', 14, 'FontWeight', 'bold');
    grid(ax_topo_enova, 'on'); box(ax_topo_enova, 'off');
end

% Preserve colormaps across all panels
colormap(ax1, parula);
if exist('bg_cmap', 'var') && ~isempty(bg_cmap)
    colormap(ax2, bg_cmap);
end
colormap(ax_topo_enova, 'turbo');

metrics = struct('ssi_before_mean', mean(ssi_before), ...
                 'ssi_after_mean', mean(ssi_after), ...
                 'ssi_artifacts_mean', mean(ssi_artifacts), ...
                 'lda_accuracy', lda_accuracy, ...
                 'signal_silhouette', sil_signal, ...
                 'ideal_power_target_db', ideal_power_target, ...
                 'enova_per_channel', enova_ch, ...
                 'psd_before_db', psd_b_db, ...
                 'psd_after_db', psd_a_db, ...
                 'psd_noise_db', psd_n_db, ...
                 'psd_freqs', f_psd, ...
                 'enova_weights', w_ch);

% --- Enforce title fonts after topoplot to prevent EEGLAB font scaling override ---
all_ax = findobj(fig, 'Type', 'axes');
for k = 1:length(all_ax)
    t_obj = all_ax(k).Title;
    if ~isempty(t_obj.String)
        t_str = string(t_obj.String);
        if any(contains(t_str, ["Before Denoising", "After Denoising"]))
            set(t_obj, 'FontSize', 13, 'FontWeight', 'bold');
        end
    end
end

% --- Instant Reveal ---
set(fig, 'Color', 'w', 'InvertHardcopy', 'off');
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
    
    % Strict manual sizing for 3-panel layout
    new_w = 0.22; 
    new_h = 0.60;
    
    fig_parent = ancestor(ax, 'figure');
    set(ax, 'Position', [pos(1), pos(2), new_w, new_h]);
    ax_x = axes('Parent', fig_parent, 'Position', [pos(1), pos(2) + new_h + margin, new_w, 0.080], 'Units', 'normalized', 'Color', 'none', 'XLim', ax.XLim, 'XTick', [], 'YTick', [], 'XAxisLocation', 'bottom', 'Box', 'off');
    hold(ax_x, 'on'); title(ax_x, ttl, 'FontSize', 20, 'FontWeight', 'bold');
    ax_y = axes('Parent', fig_parent, 'Position', [pos(1) + new_w + margin, pos(2), 0.025, new_h], 'Units', 'normalized', 'Color', 'none', 'YLim', ax.YLim, 'YTick', [], 'XTick', [], 'YAxisLocation', 'left', 'Box', 'off');
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
function [pxx, freqs] = SENSAI_compute_welch_psd(data, srate, nfft, w, noverlap)
    n_chans = size(data, 1);
    step = nfft - noverlap;
    n_pnts = size(data, 2);
    starts = 1:step:(n_pnts - nfft + 1);
    if isempty(starts), starts = 1; data = [data, zeros(n_chans, nfft - n_pnts)]; end
    n_segs = length(starts);
    w_norm = sum(w.^2);
    w_mat = repmat(w', n_chans, 1);
    acc = zeros(nfft/2 + 1, n_chans);
    for s_idx = 1:n_segs
        seg = double(data(:, starts(s_idx):starts(s_idx)+nfft-1));
        seg = (seg - mean(seg, 2)) .* w_mat;
        fft_seg = fft(seg, nfft, 2);
        p = (abs(fft_seg(:, 1:nfft/2+1)).^2) / (srate * w_norm);
        p(:, 2:end-1) = 2 * p(:, 2:end-1);
        acc = acc + p';
    end
    pxx = acc / n_segs;
    freqs = (0:nfft/2)' * (srate / nfft);
end
