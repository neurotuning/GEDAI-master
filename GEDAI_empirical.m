function [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band] = GEDAI_empirical(EEGin, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, ENOVA_threshold, signal_type)
% GEDAI_empirical - First identifies high-fidelity EEG epochs based on SSI & GFP metrics, 
%                   then uses their Log-Euclidean Mean covariance as a custom reference 
%                   matrix to run the standard GEDAI denoising pipeline.
%
% Usage:
%   [EEGclean] = GEDAI_empirical(EEGin); % Uses all default values
%
%   [EEGclean, EEGartifacts, SENSAI_score, etc.] = GEDAI_empirical(EEGin, 'auto', 12, 0.5, 'precomputed');
%
% Inputs: 
%   (All inputs match the standard GEDAI.m function)
%
%   EEGin                       - EEG data in EEGLab format
%   artifact_threshold_type     - "auto-", "auto" or "auto+". Default is "auto".
%   epoch_size_in_cycles        - Wave cycles per wavelet band. Default is 12.
%   lowcut_frequency            - exclude bands below this frequency (Hz). Default is 0.5 Hz.
%   ref_matrix_type             - Matrix used for the initial "empirical" selection. 
%                                 Default is "auto".
%   parallel                    - Boolean for parallel processing. Default is true.
%   visualize_artifacts         - Boolean for artifact visualization. Default is false.
%   ENOVA_threshold             - Threshold for rejecting epochs. Default is inf.
%   signal_type                 - 'eeg' or 'meg'. Default is 'eeg'.
%
% Outputs:
%   (All outputs match the standard GEDAI.m function)

% Handle input arguments matching GEDAI.m defaults
if nargin < 2 || isempty(artifact_threshold_type), artifact_threshold_type = 'auto'; end
if nargin < 3 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 4 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 5 || isempty(ref_matrix_type), ref_matrix_type = 'auto'; end
if nargin < 6 || isempty(parallel), parallel = true; end
if nargin < 7 || isempty(visualize_artifacts), visualize_artifacts = false; end
if nargin < 8 || isempty(ENOVA_threshold), ENOVA_threshold = inf; end
if nargin < 9 || isempty(signal_type), signal_type = 'eeg'; end

% Handle internal min_ENOVA_per_epoch and ref_matrix_type mapping
if ischar(ref_matrix_type) && ismember(ref_matrix_type, {'auto', 'empirical'})
    if strcmp(ref_matrix_type, 'auto')
        min_ENOVA_per_epoch = 0.1;
    else
        min_ENOVA_per_epoch = inf;
    end
    ref_matrix_type_internal = 'precomputed';
else
    min_ENOVA_per_epoch = inf;
    ref_matrix_type_internal = ref_matrix_type;
end

p = fileparts(which('GEDAI'));
addpath(fullfile(p, 'auxiliaries'));

%% --- PHASE 0: Sanity-Check if High-Quality Epochs Exist ---
disp([newline 'GEDAI_empirical: PHASE 0 - Sanity-Check if High-Quality Epochs Exist...']);
% Run standard GEDAI with precomputed leadfield as baseline
[~, ~, ~, ~, ~, ~, ENOVA0] = GEDAI(EEGin, 'auto', epoch_size_in_cycles, lowcut_frequency, 'precomputed', parallel, false, inf, signal_type, false);

hq_mask = ENOVA0 <= min_ENOVA_per_epoch;
n_hq = sum(hq_mask);
fprintf('GEDAI_empirical: PHASE 0 - Found %d epochs with ENOVA <= %.2f\n', n_hq, min_ENOVA_per_epoch);

if n_hq < 3
    warning('GEDAI_empirical: Sanity check failed! Only %d epochs meet the min_ENOVA_per_epoch threshold.', n_hq);
end

%% --- PHASE 1: Empirical High-Quality Epoch Selection ---
disp([newline 'GEDAI_empirical: PHASE 1 - Empirical High-Quality Epoch Selection...']);

% 1. Pre-processing for empirical phase
% Average referencing
if strcmp(lower(signal_type), 'eeg')
    EEGemp = GEDAI_nonRankDeficientAveRef(EEGin);
else
    EEGemp = EEGin;
end

% High-pass wavelet-based filtering (0.1 Hz baseline for empirical phase)
hp_freq = 0.1;
hp_levels = ceil(log2(EEGemp.srate / hp_freq) - 1);
max_lev = floor(log2(size(EEGemp.data, 2)));
hp_levels = min(hp_levels, max_lev);
hp_levels = max(hp_levels, 3);
wavelet_type = 'haar';

try
    data_in = double(EEGemp.data');
    wpt_hp = modwt_custom(data_in, wavelet_type, hp_levels);
    mra_hp = modwtmra_custom(wpt_hp, wavelet_type);
    clear wpt_hp;
catch
    wpt_hp = modwt(double(EEGemp.data'), wavelet_type, hp_levels);
    mra_hp = modwtmra(wpt_hp, wavelet_type);
end

srate = EEGemp.srate;
num_bands_hp = size(mra_hp, 1);
upper_bounds = srate ./ (2.^(1:num_bands_hp));
bands_to_zero = find(upper_bounds <= hp_freq);
if ~isempty(bands_to_zero)
    mra_hp(bands_to_zero, :, :) = 0;
end
EEGemp.data = squeeze(sum(mra_hp, 1))';
clear mra_hp;

% 2. Get Empirical Anchor Subspace
if ischar(ref_matrix_type_internal) && strcmp(ref_matrix_type_internal, 'precomputed')
    L = load('fsavLEADFIELD_4_GEDAI.mat');
    template_labels = {L.leadfield4GEDAI.electrodes.Name};
    chanidx = zeros(1, length(EEGin.chanlocs));
    for i = 1:length(EEGin.chanlocs)
        lbl = lower(EEGin.chanlocs(i).labels);
        [found, idx] = ismember(lbl, lower(template_labels));
        if found
            chanidx(i) = idx;
        else
            for j = 1:length(template_labels)
                if contains(lbl, lower(template_labels{j}))
                    chanidx(i) = j;
                    break;
                end
            end
        end
    end
    if any(chanidx == 0), error('GEDAI_empirical: Electrode labels not matched for empirical phase.'); end
    refCOV_emp = L.leadfield4GEDAI.gram_matrix_avref(chanidx,chanidx);
else
    % If ref_matrix_type is already a matrix
    refCOV_emp = ref_matrix_type_internal;
end

% Regularize anchor
reg_lambda = 0.05;
reg_val = trace(refCOV_emp) / length(refCOV_emp);
refCOV_reg = (1-reg_lambda)*refCOV_emp + reg_lambda*reg_val*eye(length(refCOV_emp));

[Vref, Dref] = eig(refCOV_reg);
[~, idxRef] = sort(diag(Dref), 'descend');
basis_ref = Vref(:, idxRef(1:3));

% 3. Calculate SSI/GFP per 1s epoch
emp_epoch_size = 1;
samples_per_epoch = srate * emp_epoch_size;
n_epochs = floor(size(EEGemp.data, 2) / samples_per_epoch);
data_epoched = reshape(EEGemp.data(:, 1:(n_epochs*samples_per_epoch)), size(EEGemp.data,1), samples_per_epoch, []);

SSI = -inf(1, n_epochs);
mean_GFP = -inf(1, n_epochs);
epoch_covs = cell(1, n_epochs);

% Resample Phase 0 mask to match Phase 1 resolution if needed
if length(hq_mask) ~= n_epochs
    fprintf('GEDAI_empirical: Resampling Phase 0 mask (%d) to match Phase 1 epochs (%d)...\n', length(hq_mask), n_epochs);
    xq = linspace(1, length(hq_mask), n_epochs);
    hq_mask_resampled = interp1(1:length(hq_mask), double(hq_mask), xq, 'nearest') > 0.5;
else
    hq_mask_resampled = hq_mask;
end

for e = 1:n_epochs
    % Only process epochs that passed the Phase 0 sanity-check (using resampled mask)
    if hq_mask_resampled(e)
        ep_data = data_epoched(:,:,e);
        c_ep = cov(ep_data');
        epoch_covs{e} = c_ep;
        [V, D] = eig(c_ep);
        [~, s_idx] = sort(diag(D), 'descend');
        basis_ep = V(:, s_idx(1:3));
        cos_val = subspace_angles(basis_ep, basis_ref);
        SSI(e) = prod(cos_val);
        mean_GFP(e) = mean(std(ep_data, 1));
    end
end

% 4. Weighted Scoring Selection (Maximize both, Prioritize SSI)
% NEW RULE: HQ epochs must also have SSI >= 0.4
hq_ssi_mask = (SSI >= 0.4);
hq_combined_mask = hq_mask_resampled & hq_ssi_mask;
n_hq_final = sum(hq_combined_mask);

fprintf('GEDAI_empirical: PHASE 1 - Found %d epochs passing both ENOVA and SSI >= 0.4 criteria.\n', n_hq_final);

% Normalize to [0,1] range (only considering final high-quality epochs)
norm_SSI = -inf(size(SSI));
norm_GFP = -inf(size(mean_GFP));

if n_hq_final > 0
    hq_idx_list = find(hq_combined_mask);
    
    vals_SSI = SSI(hq_idx_list);
    min_SSI = min(vals_SSI); max_SSI = max(vals_SSI);
    if max_SSI > min_SSI
        norm_SSI(hq_idx_list) = (vals_SSI - min_SSI) / (max_SSI - min_SSI);
    else
        norm_SSI(hq_idx_list) = 1;
    end
    
    vals_GFP = mean_GFP(hq_idx_list);
    min_GFP = min(vals_GFP); max_GFP = max(vals_GFP);
    if max_GFP > min_GFP
        norm_GFP(hq_idx_list) = (vals_GFP - min_GFP) / (max_GFP - min_GFP);
    else
        norm_GFP(hq_idx_list) = 1;
    end
end

% Combined score: 80% SSI, 20% GFP weight
empirical_score = 0.8 * norm_SSI + 0.2 * norm_GFP;

% Select top 10% of total epochs, but limited by available HQ epochs
n_top = min(n_hq_final, max(1, round(0.10 * n_epochs)));
[~, sort_idx] = sort(empirical_score, 'descend');
high_fid_idx = sort_idx(1:n_top);

fprintf('GEDAI_empirical: Identified %d high-fidelity epochs out of %d available high-quality epochs (from %d total) using weighted scoring.\n', n_top, n_hq_final, n_epochs);

if isempty(high_fid_idx)
    warning('GEDAI_empirical: No high-quality epochs found (ENOVA and SSI criteria). Falling back to standard precomputed leadfield.');
    empiricalLogMeanCOV = 'precomputed';
else
    % Calculate Log-Euclidean Mean covariance
    disp('GEDAI_empirical: Calculating Log-Euclidean Mean of high-quality covariances...');
    log_sum_cov = zeros(size(epoch_covs{high_fid_idx(1)}));
    for i = 1:length(high_fid_idx)
        % Use logm for matrix logarithm
        log_sum_cov = log_sum_cov + logm(epoch_covs{high_fid_idx(i)});
    end
    empiricalLogMeanCOV = expm(log_sum_cov / length(high_fid_idx));
    % Ensure result is symmetric
    empiricalLogMeanCOV = (empiricalLogMeanCOV + empiricalLogMeanCOV') / 2;
end

% Visualization
if visualize_artifacts && ~ischar(empiricalLogMeanCOV)
    figure('Name', 'GEDAI Empirical Selection');
    scatter(SSI, mean_GFP, 30, empirical_score, 'filled', 'MarkerFaceAlpha', 0.4); hold on;
    if ~isempty(high_fid_idx)
        scatter(SSI(high_fid_idx), mean_GFP(high_fid_idx), 40, 'g', 'LineWidth', 1.5);
        legend('All Epochs (Color=Score)', 'Selected High Quality');
    else
        legend('All Epochs (Color=Score)');
    end
    xlabel('Subspace Similarity Index (SSI)');
    ylabel('Global Field Power (GFP)');
    title('GEDAI High-Fidelity Epoch Selection');
    grid on;
    c = colorbar;
    ylabel(c, 'Selection Score');
end

%% --- PHASE 2: Standard GEDAI Pipeline ---
if ischar(empiricalLogMeanCOV)
    disp([newline 'GEDAI_empirical: PHASE 2 - Standard GEDAI denoising using ' empiricalLogMeanCOV ' leadfield (fallback)...']);
else
    disp([newline 'GEDAI_empirical: PHASE 2 - GEDAI denoising using custom empirical matrix...']);
end

[EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band] = ...
    GEDAI(EEGin, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, empiricalLogMeanCOV, parallel, visualize_artifacts, ENOVA_threshold, signal_type, visualize_artifacts);

% Update history
com = ['% GEDAI_empirical (Phase 1 Log-Euclidean mean selection followed by Phase 2 denoising)' newline com];

end
