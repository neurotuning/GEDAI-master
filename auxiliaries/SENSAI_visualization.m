function SENSAI_visualization(ref_cov, C_before, C_after, C_artifacts)
% SENSAI_VISUALIZATION Classify empirical covariance matrices against a Leadfield
%
% Inputs:
%   ref_cov     - Leadfield covariance matrix (Channels x Channels)
%   C_before    - Cell array of empirical covariance matrices BEFORE cleaning
%   C_after     - Cell array of empirical covariance matrices AFTER cleaning
%   C_artifacts - Cell array of empirical covariance matrices of removed ARTIFACTS

if ~iscell(C_before)
    num_emp = size(C_before, 3);
    temp_before = cell(num_emp, 1);
    temp_after = cell(num_emp, 1);
    temp_artifacts = cell(num_emp, 1);
    for i=1:num_emp
        temp_before{i} = C_before(:,:,i);
        temp_after{i} = C_after(:,:,i);
        temp_artifacts{i} = C_artifacts(:,:,i);
    end
    C_before = temp_before;
    C_after = temp_after;
    C_artifacts = temp_artifacts;
else
    num_emp = length(C_before);
end

%% 1. Generate the "Neural Continent" (Single Reference Point)
% The reference point is the overarching Gram matrix of the entire leadfield.
num_channels = size(ref_cov, 1);

%% 2. Calculate 3D Principal Angles against Reference Subspace
%disp('Computing 3D Principal Angles against Leadfield...');
SSI_top_PCs = 3;

% Ensure we limit to number of channels if very small
if SSI_top_PCs > num_channels
    SSI_top_PCs = num_channels;
end

% Reference subspace
[Vref, Dref] = eig(ref_cov);
[~, idx] = sort(diag(Dref), 'descend');
basis_ref = Vref(:, idx(1:SSI_top_PCs));

angs_before = extract_angles(C_before, basis_ref, SSI_top_PCs);
angs_after = extract_angles(C_after, basis_ref, SSI_top_PCs);
angs_artifacts = extract_angles(C_artifacts, basis_ref, SSI_top_PCs);

%% 3. Native MATLAB Scatter Visualizations
% 3D Principal Angles Scatter Plot
plot_3d_angles(angs_before, angs_after, angs_artifacts);

<<<<<<< Updated upstream
end

=======
%% ── 3. Epoch power  (log10 of trace) ────────────────────────────────────
lpow_before    = 10 * log10(extract_power(C_before));
lpow_after     = 10 * log10(extract_power(C_after));
pow_art        = extract_power(C_artifacts);
valid_art      = pow_art > eps;
ssi_artifacts  = ssi_artifacts(valid_art);
lpow_artifacts = 10 * log10(pow_art(valid_art));

% Ideal Target: 100% Subspace Alignment at current signal power
ideal_power_target = median(lpow_after);

%% ── 4. 2D LDA on [SSI, log-power] ──────────────────────────────────────
X_lda = [ssi_after(:),     lpow_after(:); ...
         ssi_artifacts(:), lpow_artifacts(:)];
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
ylabel(ax1, 'SSI  (geom. mean of top-3 PC cosines)', 'FontSize', 11);
title(ax1, sprintf('Before GEDAI\nn = %d epochs (50%% overlapping)  |  Mean SSI = %.3f', ...
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
if numel(lpow_artifacts) > 2
    draw_ellipse(ax2, lpow_artifacts, ssi_artifacts, col_noise, 0.95);
end

% Calculate Visual Silhouette using min-max normalized axes
if isempty(lpow_artifacts)

    s_i_br = 0;
else
    p_min = min([lpow_after; lpow_artifacts]);
    p_max = max([lpow_after; lpow_artifacts]);
    p_range = max(1e-6, p_max - p_min);

    X_sig_br = [ssi_after, (lpow_after - p_min) / p_range];
    X_noise_br = [ssi_artifacts, (lpow_artifacts - p_min) / p_range];

    D_sig_br = sqrt(max(0, bsxfun(@plus, sum(X_sig_br.^2, 2), sum(X_sig_br.^2, 2)') - 2*(X_sig_br*X_sig_br')));
    a_i_br = (sum(D_sig_br, 2)) ./ max(1, numel(ssi_after) - 1);

    D_bg_br = sqrt(max(0, bsxfun(@plus, sum(X_sig_br.^2, 2), sum(X_noise_br.^2, 2)') - 2*(X_sig_br*X_noise_br')));
    b_i_br = mean(D_bg_br, 2);

    s_i_br = mean((b_i_br - a_i_br) ./ max(a_i_br, b_i_br));
end
if ~isnan(lda_accuracy)
    ttl = sprintf('After GEDAI  |  2D LDA accuracy: %.1f%%\nMean SSSI: %.3f   |   Mean NSSI: %.3f   |   Sil: %.2f', ...
                  lda_accuracy, mean(ssi_after), mean(ssi_artifacts), s_i_br);
else
    ttl = sprintf('After GEDAI\nMean SSSI: %.3f   |   Mean NSSI: %.3f   |   Sil: %.2f', ...
                  mean(ssi_after), mean(ssi_artifacts), s_i_br);
end
title(ax2, ttl, 'FontSize', 11);

xlabel(ax2, 'Epoch Power (dB)',                'FontSize', 11);
ylabel(ax2, 'SSI  (geom. mean of top-3 PC cosines)', 'FontSize', 11);
legend(ax2, [h_star, h_sig, h_noise], ...
       {'Target Subspace', ...
        sprintf('Signal  (mean SSI=%.3f)', mean(ssi_after)), ...
        sprintf('Noise   (mean SSI=%.3f)', mean(ssi_artifacts))}, ...
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
end


%% ══════════════════════════════════════════════════════════════════════════
>>>>>>> Stashed changes
function angs = extract_angles(C_array, basis_ref, top_PCs)
    num_emp = length(C_array);
    angs = zeros(num_emp, top_PCs);
    for i = 1:num_emp
        [V, D] = eig(C_array{i});
        [~, idx] = sort(diag(D), 'descend');
        basis_c = V(:, idx(1:top_PCs));
        cos_theta = subspace_angles(basis_c, basis_ref);
        angs(i, :) = cos_theta(:)';
    end
end

function plot_3d_angles(angs_before, angs_after, angs_artifacts)
    figure('Name', 'GEDAI 3D Principal Subspace Angles', 'Color', 'w', 'Position', [150 150 1200 600]);
    
    % Subplot 1: Before Denoising (to match plot_manifold style)
    subplot(1, 2, 1);
    ssi_before = prod(angs_before, 2);
    [~, sort_idx] = sort(ssi_before, 'ascend'); % Sort so high similarity is plotted last/on top
    scatter3(angs_before(sort_idx,1), angs_before(sort_idx,2), angs_before(sort_idx,3), ...
             60, ssi_before(sort_idx), 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.8);
    hold on;
    scatter3(1, 1, 1, 300, 'yellow', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
    
    colormap(parula);
    c = colorbar;
    c.Label.String = 'Subspace Similarity Index (SSI)';
    c.Label.FontSize = 13;
    caxis([0 1]);
    
    xlabel('SSI \bf{PC_1}'); ylabel('SSI \bf{PC_2}'); zlabel('SSI \bf{PC_3}');
    title(sprintf('Before GEDAI: \nMean SSI: %.3f', mean(ssi_before)));
    grid on; view(45, 30);
    xlim([0 1]); ylim([0 1]); zlim([0 1]);
    
    % Subplot 2: After Denoising & Artifacts
    subplot(1, 2, 2);
    hold on;
    % 1. Artifact Epochs (Red)
    h_artifact = scatter3(angs_artifacts(:,1), angs_artifacts(:,2), angs_artifacts(:,3), ...
             60, [0.8 0.1 0.1], 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.6);
    % 2. Cleaned Epochs (Green)
    h_clean = scatter3(angs_after(:,1), angs_after(:,2), angs_after(:,3), ...
             60, [0.1 0.8 0.1], 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.8);
    % 3. Leadfield Subspace (Yellow Star)
    h_star = scatter3(1, 1, 1, 300, 'yellow', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
    
    % Calculate 5-fold LDA Accuracy
    X_lda = [angs_after; angs_artifacts];
    Y_lda = [ones(size(angs_after, 1), 1); zeros(size(angs_artifacts, 1), 1)]; % 1 = Clean, 0 = Artifact
    lda_model = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5);
    lda_loss = kfoldLoss(lda_model);
    lda_accuracy = (1 - lda_loss) * 100;
    
    ssi_after = prod(angs_after, 2);
    nssi_after = prod(angs_artifacts, 2);
    
    xlabel('SSI \bf{PC_1}'); ylabel('SSI \bf{PC_2}'); zlabel('SSI \bf{PC_3}');
    title(sprintf('After GEDAI: \\color[rgb]{0,0.7,0}Signal \\color{black}vs \\color{red}Noise \\color{black}Epochs \n\\color{black}Mean SSSI: %.3f | NSSI: %.3f  | Accuracy: %.1f%%', mean(ssi_after), mean(nssi_after), lda_accuracy));
    legend([h_star, h_clean, h_artifact], {'Leadfield Subspace', 'Cleaned Signal (SSSI)', 'Removed Noise (NSSI)'}, 'Location', 'best');
    grid on; view(45, 30);
    xlim([0 1]); ylim([0 1]); zlim([0 1]);
    
    sgtitle('Signal & Noise Subspace Similarity Index (SENSAI) per epoch');

end