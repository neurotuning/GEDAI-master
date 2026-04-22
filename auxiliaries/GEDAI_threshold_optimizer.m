% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% Copyright (C) [2025] Tomas Ros & Abele Michela — NeuroTuning Lab

function [optimal_threshold, optimization_results] = GEDAI_threshold_optimizer(eeg_data, srate, chanlocs, refCOV, epoch_size_in_cycles, lowcut_frequency, signal_type, n_evals, use_parallel, visualize)
% GEDAI_THRESHOLD_OPTIMIZER  Bayesian optimisation of the FULL GEDAI pipeline.
%
%   Optimises the threshold [0, 10] by running the entire multi-band 
%   reconstruction for each evaluation and maximising the final SSSI-NSSI.

if nargin < 5 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 6 || isempty(lowcut_frequency),     lowcut_frequency = 0.5; end
if nargin < 7 || isempty(signal_type),          signal_type = 'eeg'; end
if nargin < 8 || isempty(n_evals),              n_evals = 15; end
if nargin < 9 || isempty(use_parallel),         use_parallel = false; end
if nargin < 10 || isempty(visualize),           visualize = true; end

% Fixed pipeline parameters (must match GEDAI.m)
broadband_epoch_size = 1; 
num_bands = 9;
wavelet_type = 'haar';

%% Bayesian optimisation ---------------------------------------------------
objective_fn = @(p) eval_full_pipeline(p.threshold, eeg_data, srate, chanlocs, refCOV, ...
    epoch_size_in_cycles, lowcut_frequency, signal_type, broadband_epoch_size, num_bands, wavelet_type);

plot_fcn = {};
if visualize, plot_fcn = @plotObjectiveModel; end

bo_results = bayesopt(objective_fn, ...
    optimizableVariable('threshold', [0, 10], 'Type', 'real'), ...
    'MaxObjectiveEvaluations',  n_evals, ...
    'UseParallel',              false, ...
    'IsObjectiveDeterministic', false, ...
    'Verbose',                  0, ...
    'PlotFcn',                  plot_fcn, ...
    'OutputFcn',                @assignInBase, ...
    'SaveVariableName',         'GEDAI_bayesian_optimization');

%% Final results -----------------------------------------------------------
optimal_threshold = max(0, min(10, bo_results.XAtMinObjective.threshold));

optimization_results = struct('bo_results', bo_results, ...
    'optimal_threshold', optimal_threshold, ...
    'best_sssi_nssi_score', -bo_results.MinObjective, ...
    'n_evals',           n_evals);
end


%% ========================================================================
%  Objective function: FULL PIPELINE RECONSTRUCTION
%% ========================================================================
function neg_score = eval_full_pipeline(threshold, eeg_data, srate, chanlocs, refCOV, ...
    epoch_size_in_cycles, lowcut_frequency, signal_type, bb_epoch_size, num_bands, wavelet_type)

try
    % 1. Broadband Denoising
    [cleaned_bb, ~] = GEDAI_per_band(double(eeg_data), srate, chanlocs, ...
        threshold, bb_epoch_size, refCOV, 'parabolic', false, signal_type, 0, 12);
    
    % 2. Multi-band cleaning
    % We replicate the frequency-dependent logic from GEDAI.m
    lower_freqs = srate ./ (2.^( (1:num_bands) + 1 ));
    upper_freqs = srate ./ (2.^( 1:num_bands ));
    
    % Wavelet bands to process
    bands_to_process = find(upper_freqs > lowcut_frequency);
    
    % Accumulator for reconstruction
    reconstructed = zeros(size(eeg_data));
    unfiltered_data_trans = cleaned_bb'; % Match GEDAI.m orientation
    
    for f = bands_to_process
        % Extract band
        band_data = modwt_single_band(unfiltered_data_trans, wavelet_type, num_bands-1, f)';
        
        % Band-specific epoch size
        cur_epoch_size = epoch_size_in_cycles / lower_freqs(f);
        
        % Band-specific minThreshold (alpha/beta protection)
        cur_center_freq = (lower_freqs(f) + upper_freqs(f)) / 2;
        cur_minThreshold = 0;
        if (cur_center_freq >= 7 && cur_center_freq <= 13), cur_minThreshold = -6; end
        
        % Clean band
        [cleaned_band, ~] = GEDAI_per_band(double(band_data), srate, chanlocs, ...
            threshold, cur_epoch_size, refCOV, 'parabolic', false, signal_type, cur_minThreshold);
        
        % Sum to reconstruction
        reconstructed = reconstructed + cleaned_band;
    end
    
    % 3. Scoring
    artifacts = eeg_data - reconstructed;
    % Use bb_epoch_size for final scoring consistency
    noise_multiplier=2; % For broadband components, apply lower weight (default=2) to noise variance for SENSAI scoring
    score = SENSAI_basic(reconstructed, artifacts, srate, bb_epoch_size, refCOV, noise_multiplier, signal_type);
    
    if isnan(score) || ~isfinite(score), neg_score = 0; else, neg_score = -score; end
    
catch ME
    warning('GEDAI_threshold_optimizer:evalError', 'threshold=%.2f: %s', threshold, ME.message);
    neg_score = 0;
end
end
