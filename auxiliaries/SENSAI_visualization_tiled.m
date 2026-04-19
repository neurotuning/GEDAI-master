function SENSAI_visualization_tiled(refCOV, viz_data_array, band_labels)
% SENSAI_VISUALIZATION_TILED  Multi-panel 2D diagnostic for all wavelet bands
%
% Displays SSI composite (Alignment) vs Epoch Power (dB) for every band.
% Green = Clean Signal cluster, Red = Removed Artifact cluster.

num_bands = length(viz_data_array);
if num_bands == 0, return; end

% Determine grid layout
rows = ceil(sqrt(num_bands + 1)); % +1 for padding/balance
cols = ceil(num_bands / rows);

figure('Name', 'Multi-band SENSAI Diagnostic', 'Color', 'w', 'Position', [100 100 1400 900]);

if exist('tiledlayout', 'builtin')
    t = tiledlayout(rows, cols, 'TileSpacing', 'Compact', 'Padding', 'Normal');
end

% Color palette
col_sig   = [0.08 0.72 0.22];
col_noise = [0.85 0.13 0.13];
col_star  = [1.00 0.88 0.00];

for i = 1:num_bands
    if exist('tiledlayout', 'builtin')
        nexttile(t);
    else
        subplot(rows, cols, i);
    end
    
    data = viz_data_array{i};
    if isempty(data), continue; end
    
    hold on;
    % 1. Artifacts (Red)
    scatter(data.lpow_artifacts, data.ssi_artifacts, 15, col_noise, 'filled', 'MarkerFaceAlpha', 0.35);
    % 2. Signal (Green)
    scatter(data.lpow_after, data.ssi_after, 15, col_sig, 'filled', 'MarkerFaceAlpha', 0.55);
    % 3. Target (Yellow Star at SSI=1, median power)
    ideal_p = median(data.lpow_after);
    scatter(ideal_p, 1, 120, col_star, 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    % Calculate 5-fold LDA Accuracy for this specific band
    % Jitter to prevent singular matrices in degenerate cases
    X_lda = [data.ssi_after(:), data.lpow_after(:); data.ssi_artifacts(:), data.lpow_artifacts(:)];
    X_lda = X_lda + 1e-6 * randn(size(X_lda)); 
    Y_lda = [ones(numel(data.ssi_after), 1); zeros(numel(data.ssi_artifacts), 1)];
    try
        lda_mdl = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5, 'DiscrimType', 'pseudoLinear');
        acc = (1 - kfoldLoss(lda_mdl)) * 100;
    catch
        acc = NaN;
    end
    
    % Title with metrics
    if isnan(acc)
        title_str = sprintf('%s\nS:%.2f | N:%.2f | Sil:%.2f', band_labels{i}, mean(data.ssi_after), mean(data.ssi_artifacts), data.silhouette);
    else
        title_str = sprintf('%s (Acc:%.0f%%)\nS:%.2f | N:%.2f | Sil:%.2f', band_labels{i}, acc, mean(data.ssi_after), mean(data.ssi_artifacts), data.silhouette);
    end
    title(title_str, 'FontSize', 9);
    
    % Style
    grid on; ax = gca; ax.GridAlpha = 0.15;
    ylim([-0.05 1.15]);
    if i > (num_bands - cols), xlabel('Power (dB)', 'FontSize', 8); end
    if mod(i-1, cols) == 0, ylabel('SSI', 'FontSize', 8); end
    set(ax, 'FontSize', 8);
end

if exist('tiledlayout', 'builtin')
    title(t, 'SENSAI Subspace Partitioning (All Bands)', 'FontSize', 14, 'FontWeight', 'bold');
else
    sgtitle('SENSAI Subspace Partitioning (All Bands)');
end

end
