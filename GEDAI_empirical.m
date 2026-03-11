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
%                                 Default is "precomputed".
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
if nargin < 5 || isempty(ref_matrix_type), ref_matrix_type = 'precomputed'; end
if nargin < 6 || isempty(parallel), parallel = true; end
if nargin < 7 || isempty(visualize_artifacts), visualize_artifacts = false; end
if nargin < 8 || isempty(ENOVA_threshold), ENOVA_threshold = inf; end
if nargin < 9 || isempty(signal_type), signal_type = 'eeg'; end

p = fileparts(which('GEDAI'));
addpath(fullfile(p, 'auxiliaries'));

%% --- PHASE 1: Empirical High-Fidelity Epoch Selection ---
disp([newline 'GEDAI_empirical: PHASE 1 - Empirical High-Fidelity Epoch Selection...']);

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
if ischar(ref_matrix_type) && strcmp(ref_matrix_type, 'precomputed')
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
    refCOV_emp = ref_matrix_type;
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

SSI = zeros(1, n_epochs);
mean_GFP = zeros(1, n_epochs);
epoch_covs = cell(1, n_epochs);

for e = 1:n_epochs
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

% 4. Weighted Scoring Selection (Maximize both, Prioritize SSI)
% Normalize to [0,1] range
min_SSI = min(SSI); max_SSI = max(SSI);
min_GFP = min(mean_GFP); max_GFP = max(mean_GFP);

if max_SSI > min_SSI
    norm_SSI = (SSI - min_SSI) / (max_SSI - min_SSI);
else
    norm_SSI = ones(size(SSI));
end

if max_GFP > min_GFP
    norm_GFP = (mean_GFP - min_GFP) / (max_GFP - min_GFP);
else
    norm_GFP = ones(size(mean_GFP));
end

% Combined score: 80% SSI, 20% GFP weight
empirical_score = 0.8 * norm_SSI + 0.2 * norm_GFP;

% Select top 10% of epochs
n_top = max(1, round(0.10 * n_epochs));
[~, sort_idx] = sort(empirical_score, 'descend');
high_fid_idx = sort_idx(1:n_top);

fprintf('GEDAI_empirical: Identified %d high-fidelity epochs (Top 10%%) out of %d using weighted scoring (80%% SSI, 20%% GFP).\n', n_top, n_epochs);

if isempty(high_fid_idx)
    error('GEDAI_empirical: Selection failed. No epochs identified.');
end

% Calculate Log-Euclidean Mean covariance
disp('GEDAI_empirical: Calculating Log-Euclidean Mean of high-fidelity covariances...');
log_sum_cov = zeros(size(epoch_covs{1}));
for i = 1:length(high_fid_idx)
    % Use logm for matrix logarithm
    log_sum_cov = log_sum_cov + logm(epoch_covs{high_fid_idx(i)});
end
empiricalLogMeanCOV = expm(log_sum_cov / length(high_fid_idx));
% Ensure result is symmetric
empiricalLogMeanCOV = (empiricalLogMeanCOV + empiricalLogMeanCOV') / 2;

% Visualization
if visualize_artifacts
    figure('Name', 'GEDAI Empirical Selection');
    scatter(SSI, mean_GFP, 30, empirical_score, 'filled', 'MarkerFaceAlpha', 0.4); hold on;
    scatter(SSI(high_fid_idx), mean_GFP(high_fid_idx), 40, 'g', 'LineWidth', 1.5);
    colorbar;
    xlabel('SSI (Alignment with Leadfield)'); 
    ylabel('Mean GFP (Signal Power)'); 
    title('Empirical Epoch Selection (Weighted: 80% SSI, 20% GFP)');
    legend('All Epochs (Color=Score)', 'Selected High Fidelity'); grid on;
end

%% --- PHASE 2: Standard GEDAI Pipeline ---
disp([newline 'GEDAI_empirical: PHASE 2 - Standard GEDAI denoising using empirical matrix...']);

[EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band] = ...
    GEDAI(EEGin, artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, empiricalLogMeanCOV, parallel, visualize_artifacts, ENOVA_threshold, signal_type, visualize_artifacts);

% Update history
com = ['% GEDAI_empirical (Phase 1 Log-Euclidean mean selection followed by Phase 2 denoising)' newline com];

end
