function SENSAI_visualization_tiled(ref_cov, band_SSI_angles, band_labels)
    % SENSAI_VISUALIZATION_TILED Create a tiled figure of SENSAI plots for multiple bands
    %
    % Inputs:
    %   ref_cov          - Leadfield covariance matrix
    %   band_SSI_angles  - Cell array of structs, each with angs_before, angs_after, angs_artifacts
    %   band_labels      - Cell array of strings for titles
    
    num_bands = length(band_SSI_angles);
    if num_bands == 0, return; end
    
    % Determine grid size
    cols = ceil(sqrt(num_bands));
    rows = ceil(num_bands / cols);
    
    figure('Name', 'GEDAI Tiled SENSAI Visualization', 'Color', 'w', 'Position', [100 100 300*cols 300*rows]);
    
    % Check if tiledlayout is supported (MATLAB R2019b+)
    if exist('tiledlayout', 'builtin')
        t = tiledlayout(rows, cols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    end
    
    for i = 1:num_bands
        if exist('tiledlayout', 'builtin')
            nexttile;
        else
            subplot(rows, cols, i);
        end
        
        SSI_angles = band_SSI_angles{i};
        if isempty(SSI_angles), continue; end
        
        angs_after = SSI_angles.angs_after;
        angs_artifacts = SSI_angles.angs_artifacts;
        
        hold on;
        % 1. Artifact Epochs (Red)
        scatter3(angs_artifacts(:,1), angs_artifacts(:,2), angs_artifacts(:,3), ...
                 25, [0.8 0.1 0.1], 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.4);
        % 2. Cleaned Epochs (Green)
        scatter3(angs_after(:,1), angs_after(:,2), angs_after(:,3), ...
                 25, [0.1 0.8 0.1], 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.6);
        % 3. Leadfield Subspace (Yellow Star)
        scatter3(1, 1, 1, 120, 'yellow', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        
        ssi_after = prod(angs_after, 2);
        nssi_val = prod(angs_artifacts, 2);
        
        % Check for origin points (epochs with no artifacts removed)
        is_origin = all(angs_artifacts == 0, 2);
        num_cleaned = sum(~is_origin);
        num_total = length(is_origin);
        
        % Add small amount of jitter to origin points so they are visible as a group near [0,0,0]
        if any(is_origin)
             % Tiny jitter in [0, 0.05] range
             angs_artifacts(is_origin, :) = angs_artifacts(is_origin, :) + 0.05 * rand(sum(is_origin), 3);
        end

        % Calculate 5-fold LDA Accuracy for this band
        try
            X_lda = [angs_after; angs_artifacts];
            Y_lda = [ones(size(angs_after, 1), 1); zeros(size(angs_artifacts, 1), 1)]; 
            lda_model = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5);
            lda_loss = kfoldLoss(lda_model);
            lda_accuracy = (1 - lda_loss) * 100;
        catch
            lda_accuracy = 0; % Fallback if fits fail (e.g. too few points)
        end
        
        title(sprintf('%s\nS:%.2f | N:%.2f | Acc:%.0f%% | Art:%d/%d', ...
            band_labels{i}, mean(ssi_after), mean(nssi_val), lda_accuracy, num_cleaned, num_total), 'FontSize', 9);
        grid on; view(45, 30);
        xlim([0 1]); ylim([0 1]); zlim([0 1]);
        set(gca, 'FontSize', 8);
    end
    
    if exist('tiledlayout', 'builtin')
        title(t, 'SENSAI Signal (Green) vs Noise (Red) per Wavelet Band', 'FontSize', 14, 'FontWeight', 'bold');
    else
        % sgtitle('SENSAI Signal (Green) vs Noise (Red) per Wavelet Band');
    end
end
