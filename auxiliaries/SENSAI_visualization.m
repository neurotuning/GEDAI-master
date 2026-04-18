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

end

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