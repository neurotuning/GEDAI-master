function [best_scale, best_SENSAI, opt_results, selected_threshold_type] = GEDAI_bayesopt_strength(data, srate, chanlocs, ref_matrix, max_evals, verbose, plot_opt, signal_type, epoch_size_in_cycles, lowcut_frequency, parallelize)
% GEDAI_BAYESOPT_STRENGTH - Bayesian Optimization (bayesopt) over the auto_scale
% denoising multiplier in [0.1, 5.0] (from 0.1x most aggressive to 5.0x least
% aggressive relative to 'auto' = 1.0x), preserving per-band Z3 spectral-spatial
% adaptation across Pass 1 (Broadband) and Pass 2 (Wavelets).
%
% Outer evaluation uses SENSAI_basic with noise_multiplier = local_compute_z3_noise_multiplier(...)
% computed once on the input recording (without auto_scale). After optimization,
% the final GEDAI pass evaluates SENSAI_basic with noise_multiplier = 1.
%
% Inputs:
%   data                    - (nbchan x pnts) double EEG/MEG sensor data
%   srate                   - Sampling rate in Hz
%   chanlocs                - EEG.chanlocs struct
%   ref_matrix              - Precomputed string ('auto', 'precomputed', 'interpolated', 'projection')
%                             or numeric reference covariance matrix (nbchan x nbchan)
%   max_evals               - Maximum objective evaluations (default = 12)
%   verbose                 - 1 to print log, 0 for quiet (default = 1)
%   plot_opt                - Boolean to show optimization progress plot (default = false)
%   signal_type             - 'eeg' or 'meg' (default = 'eeg')
%   epoch_size_in_cycles    - Epoch size in cycles (default = 12)
%   lowcut_frequency        - Low-cut frequency in Hz (default = 0.5)
%   parallelize             - Boolean to run each GEDAI evaluation in parallel (default = false)
%
% Outputs:
%   best_scale              - Optimal auto_scale multiplier in [0.1, 5.0] (1.0 = 'auto')
%   best_SENSAI             - Outer SENSAI_basic score (%) at the selected multiplier
%   opt_results             - Optimization results struct (including bayesopt object)
%   selected_threshold_type - 'auto:<best_scale>' if it beats 'auto', else 'auto'

if nargin < 5 || isempty(max_evals), max_evals = 12; end
if nargin < 6 || isempty(verbose), verbose = 1; end
if nargin < 7 || isempty(plot_opt), plot_opt = false; end
if nargin < 8 || isempty(signal_type), signal_type = 'eeg'; end
if nargin < 9 || isempty(epoch_size_in_cycles), epoch_size_in_cycles = 12; end
if nargin < 10 || isempty(lowcut_frequency), lowcut_frequency = 0.5; end
if nargin < 11 || isempty(parallelize), parallelize = false; end

data = double(data);
[nbchan, pnts] = size(data);

% Subsample up to 20 seconds using 3 representative blocks (early/mid/late)
% with cosine-tapered seams for fast evaluation on long recordings
max_opt_sec = 20;
max_opt_pnts = min(pnts, round(max_opt_sec * srate));
if pnts > round(max_opt_pnts * 1.25)
    blk_len = floor(max_opt_pnts / 3);
    idx1 = 1:blk_len;
    mid_start = max(1, round(pnts / 2) - floor(blk_len / 2));
    idx2 = mid_start:(mid_start + blk_len - 1);
    idx3 = (pnts - blk_len + 1):pnts;
    data_sub = stitch_blocks_cosine(data(:, idx1), data(:, idx2), data(:, idx3), min(round(0.25 * srate), floor(blk_len / 8)));
else
    data_sub = data(:, 1:max_opt_pnts);
end

EEG_eval = struct();
EEG_eval.data = data_sub;
EEG_eval.srate = srate;
EEG_eval.chanlocs = chanlocs;
EEG_eval.nbchan = nbchan;
EEG_eval.pnts = size(data_sub, 2);
EEG_eval.trials = 1;
EEG_eval.xmin = 0;
EEG_eval.xmax = (EEG_eval.pnts - 1) / srate;
EEG_eval.etc = struct('GEDAI_use_cached_invariants', true);

if verbose
    fprintf('\n=== GEDAI: Bayesian Optimization (bayesopt) of Denoising Strength ===\n');
    fprintf('  Sweeping auto_scale multiplier in [0.10x (most aggressive) .. 5.00x (least aggressive)]\n');
    fprintf('  Baseline anchor: ''auto'' (0.50x) | ''auto+'' (0.33x) | ''auto-'' (1.00x)\n');
end

warn_state = warning('off');
clean_up = onCleanup(@() warning(warn_state));

% Step 0: Evaluate 'auto' (0.50x) baseline first and compute fixed outer Z3 noise_multiplier (without auto_scale)
outer_noise_multiplier = [];
[sensai_auto, en_auto, outer_noise_multiplier, w_dirty_rec, w_rank_rec, w_hi_meff_rec] = eval_autoscale_metrics(0.5, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize, outer_noise_multiplier);
en_margin_clean = 0.05 + 2.0 * w_rank_rec;
en_cap_clean    = en_auto + en_margin_clean;
en_cap_dirty    = max(55.0, en_auto + 12.0);
en_cap          = (1.0 - w_dirty_rec) * en_cap_clean + w_dirty_rec * en_cap_dirty;

eval_hist_scale  = 0.5;
eval_hist_sensai = sensai_auto;
eval_hist_enova  = en_auto;
eval_hist_obj    = sensai_auto; % At 0.50x ('auto'), ENOVA penalty is 0

% Initial log-spaced anchor seeds across the trust region around 'auto' (0.50x)
default_seeds = [0.20; 1/3; 0.42; 0.65; 1.00; 1.50];
if verbose == 0 && ~plot_opt
    default_seeds = [0.20; 1/3; 0.42; 0.65; 1.00];
end
n_init = min(numel(default_seeds), max(1, max_evals - 1));
init_scales = default_seeds(1:n_init);

rem_budget = max(1, max_evals - 1);
use_bayesopt = (verbose > 0 || plot_opt) && ( exist('bayesopt', 'file') == 2 || exist('bayesopt', 'class') == 8 ) && (rem_budget >= 4);

if use_bayesopt
    scale_var = optimizableVariable('auto_scale', [0.10, 5.00], 'Type', 'real', 'Transform', 'log');
    init_table = table(init_scales, 'VariableNames', {'auto_scale'});
    bo_obj = @(T) bayesopt_tracked_objective(T.auto_scale);
    try
        bo_res = bayesopt(bo_obj, scale_var, ...
            'MaxObjectiveEvaluations', rem_budget, ...
            'InitialX', init_table, ...
            'IsObjectiveDeterministic', true, ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'Verbose', 0, ...
            'PlotFcn', {});
    catch
        use_bayesopt = false;
    end
end

if ~use_bayesopt
    % Fast deterministic log-grid evaluation across the trust-region anchors
    for k = 1:numel(init_scales)
        bayesopt_tracked_objective(init_scales(k));
    end
    bo_res = [];
end

[best_obj_val, best_idx] = max(eval_hist_obj);
best_scale  = eval_hist_scale(best_idx);
best_SENSAI = eval_hist_sensai(best_idx);
best_ENOVA  = eval_hist_enova(best_idx);

% Monotonic baseline guarantee: if no candidate strictly beats 'auto' (0.50x), retain 'auto'
if best_obj_val > sensai_auto + 1e-6 && abs(best_scale - 0.5) > 1e-3
    selected_threshold_type = sprintf('auto:%.6f', best_scale);
else
    best_scale  = 0.5;
    best_SENSAI = sensai_auto;
    best_ENOVA  = en_auto;
    selected_threshold_type = 'auto:0.500000';
end

opt_results = struct();
opt_results.OptimalAutoScale      = best_scale;
opt_results.OuterZ3Multiplier     = outer_noise_multiplier;
opt_results.BestSENSAI            = best_SENSAI;
opt_results.BestENOVA             = best_ENOVA;
opt_results.BaselineAutoSENSAI    = sensai_auto;
opt_results.BaselineAutoENOVA     = en_auto;
opt_results.ENOVACeiling          = en_cap;
opt_results.SelectedThresholdType = selected_threshold_type;
opt_results.EvaluatedScales       = eval_hist_scale;
opt_results.EvaluatedSENSAI       = eval_hist_sensai;
opt_results.EvaluatedENOVA        = eval_hist_enova;
opt_results.EvaluatedObjective    = eval_hist_obj;
opt_results.NumEvaluations        = numel(eval_hist_scale);
opt_results.BayesOptResults       = bo_res;

if plot_opt
    figure('Color', 'w', 'Name', 'GEDAI Strength Optimization (bayesopt 0.1x - 5.0x)');
    [xs, ord] = sort(eval_hist_scale);
    ys = eval_hist_obj(ord);
    semilogx(xs, ys, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;
    yline(sensai_auto, '--k', sprintf('''auto'' (0.50x) baseline (%.2f%%)', sensai_auto), 'LineWidth', 1.2);
    semilogx(best_scale, best_obj_val, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xlabel('Denoising Multiplier auto\_scale (0.1x = most aggressive, 5.0x = least aggressive)');
    ylabel(sprintf('Outer SENSAI\\_basic Objective (%%) [Z3 mult = %.2f]', outer_noise_multiplier));
    title(sprintf('Optimal auto\\_scale = %.2fx (Outer SENSAI = %.2f%%, ENOVA = %.1f%%)', best_scale, best_SENSAI, best_ENOVA));
    grid on;
end

if verbose
    fprintf('  Outer Z3 noise_multiplier:    %.2f (fixed across all candidate evaluations)\n', outer_noise_multiplier);
    fprintf('  Baseline ''auto'' (0.50x):      Outer SENSAI = %+.2f%% | ENOVA = %.1f%% (Ceiling = %.1f%%)\n', sensai_auto, en_auto, en_cap);
    fprintf('  Selected Optimal Multiplier:  auto_scale* = %.2fx (%d evals)\n', best_scale, opt_results.NumEvaluations);
    fprintf('  Selected Outer SENSAI/ENOVA:  Outer SENSAI = %+.2f%% | ENOVA = %.1f%%\n', best_SENSAI, best_ENOVA);
    fprintf('=====================================================================\n\n');
end

    function neg_obj = bayesopt_tracked_objective(sc_val)
        sc_val = max(0.10, min(5.00, double(sc_val)));
        % Reuse cached value if already evaluated within 1e-4
        [min_diff, idx_hit] = min(abs(eval_hist_scale - sc_val));
        if min_diff < 1e-4
            neg_obj = -eval_hist_obj(idx_hit);
            return;
        end
        [sen_v, en_v] = eval_autoscale_metrics(sc_val, EEG_eval, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize, outer_noise_multiplier);
        en_norm = max(0.10, en_margin_clean * (1.0 - w_dirty_rec) + w_dirty_rec);
        w_bal_v = (1.0 - w_rank_rec) * (1.0 - w_dirty_rec) * w_hi_meff_rec * exp(-0.5 * ((sc_val - 0.65) / 0.08)^2);
        w_rel_v = ((1.0 - w_dirty_rec)^2) * max(w_rank_rec, 0.35 * (1.0 - w_hi_meff_rec)) * max(0, min(sc_val, 1.0) - 0.5);
        w_agg_v = w_dirty_rec * max(0.0, 0.50 - sc_val) / 0.30;
        w_pen_v = (1.0 - w_dirty_rec) * max(0.0, 0.50 - sc_val) / 0.10;
        en_cap_v = en_cap + 25.0 * w_bal_v;
        obj_v = sen_v + 36.0 * w_rel_v + 12.0 * w_bal_v + 15.0 * w_agg_v - 10.0 * w_pen_v - 3.0 * (max(0, en_v - en_cap_v) / max(en_norm, 2.0 * w_bal_v))^1.5;
        eval_hist_scale(end+1, 1)  = sc_val;
        eval_hist_sensai(end+1, 1) = sen_v;
        eval_hist_enova(end+1, 1)  = en_v;
        eval_hist_obj(end+1, 1)    = obj_v;
        neg_obj = -obj_v;
    end

end

function [sensai_score, enova_pct, outer_noise_mult, w_dirty, w_rank, w_hi_meff] = eval_autoscale_metrics(sc_val, EEG_in, epoch_size_in_cycles, lowcut_frequency, ref_matrix, signal_type, parallelize, outer_noise_mult)
w_dirty   = 0.0;
w_rank    = 0.0;
w_hi_meff = 0.0;
try
    mode_str = sprintf('auto:%.6f', sc_val);
    [~, EEG_clean, EEG_art] = evalc( ...
        'GEDAI(EEG_in, mode_str, epoch_size_in_cycles, lowcut_frequency, ref_matrix, parallelize, false, inf, inf, signal_type, inf, ''AvgRef'', struct(''silent'', true));');
    X_cl = double(EEG_clean.data);
    X_ar = double(EEG_art.data);
    X_in_hp = X_cl + X_ar;
    refCOV_used = EEG_clean.etc.GEDAI.refCOV;
    enova_pct = 100 * (1 - sum(X_cl(:).^2) / max(sum(X_in_hp(:).^2), 1e-15));
    if isempty(outer_noise_mult)
        if strcmpi(signal_type, 'meg')
            outer_noise_mult = 3.0;
        else
            [raw_z3, w_rank_z3, w_macro_z3, dip_frac_z3] = compute_outer_z3_noise_multiplier(X_in_hp, EEG_in.srate, refCOV_used, signal_type);
            if isfield(EEG_clean.etc.GEDAI, 'w_rank') && isfield(EEG_clean.etc.GEDAI, 'w_macro')
                w_rank  = EEG_clean.etc.GEDAI.w_rank;
                w_macro = EEG_clean.etc.GEDAI.w_macro;
            else
                w_rank  = w_rank_z3;
                w_macro = w_macro_z3;
            end
            if isfield(EEG_clean.etc.GEDAI, 'dip_frac')
                dip_frac_z3 = EEG_clean.etc.GEDAI.dip_frac;
            end
            meff_eeg_rec = raw_z3;
            if isfield(EEG_clean.etc.GEDAI, 'm_eff_eeg')
                meff_eeg_rec = EEG_clean.etc.GEDAI.m_eff_eeg;
            end
            M_d_rec    = double(min(3, EEG_in.nbchan));
            r_sub_rec  = (M_d_rec^3) / max(double(EEG_in.nbchan - 1), 1.0);
            w_ch_dof   = 1.0 ./ (1.0 + exp(-(r_sub_rec - 0.85) / 0.08));
            w_ch_dof_g = 1.0 ./ (1.0 + exp(-(r_sub_rec - 1.25) / 0.08));
            w_hi_meff  = (1.0 - w_ch_dof) * max(0.0, min(1.0, (meff_eeg_rec - (M_d_rec - 0.5)) / max(0.5 * M_d_rec, 1e-12)));
            w_dip_eog  = 1.0 ./ (1.0 + exp(-(dip_frac_z3 - 0.65) / 0.030));
            w_z3_low   = 1.0 ./ (1.0 + exp((raw_z3 - 3.35) / 0.25));
            w_en_50    = 1.0 ./ (1.0 + exp(-(enova_pct - 50.0) / 5.0));
            w_en_20    = 1.0 ./ (1.0 + exp(-(enova_pct - 20.0) / 2.0));
            w_dirty    = (1.0 - w_rank) * max(w_macro * w_en_20, w_dip_eog * max(w_z3_low, w_ch_dof) * (1.0 - w_ch_dof_g) * w_en_50);
            outer_noise_mult = max(0.15, 1.40 - 1.15 * max(w_dirty, w_rank) - 0.25 * w_hi_meff * (1.0 - w_dirty) * (1.0 - w_rank)) * raw_z3;
        end
    end
    [sensai_score, ~, ~, ~] = SENSAI_basic(X_cl, X_ar, EEG_in.srate, 1, refCOV_used, outer_noise_mult, signal_type);
    if isnan(sensai_score) || isinf(sensai_score)
        sensai_score = -100;
        enova_pct = 100;
    end
catch
    if isempty(outer_noise_mult), outer_noise_mult = 1.0; end
    sensai_score = -100;
    enova_pct = 100;
end
end

function [m_eff, w_rank, w_macro, dip_frac] = compute_outer_z3_noise_multiplier(eeg_data, srate, refCOV, signal_type)
w_rank   = 0.0;
w_macro  = 0.0;
dip_frac = 0.5;
if ndims(refCOV) == 3
    refCOV = mean(refCOV, 3);
end
refCOV = double(real(refCOV));
refCOV = (refCOV + refCOV') / 2;
N_chans = size(eeg_data, 1);
regularization_lambda = 0.05;
reg_val = trace(refCOV) / N_chans;
refCOV_reg = (1 - regularization_lambda) * refCOV + regularization_lambda * reg_val * eye(N_chans);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;

SSI_top_PCs = min(3, N_chans);
[Vs, Ds] = eig(refCOV_reg);
[~, sidx] = sort(diag(Ds), 'descend');
evecs_Template_cov = Vs(:, sidx(1:SSI_top_PCs));
evecs_lead         = Vs(:, sidx(1:min(SSI_top_PCs^2, max(1, N_chans - 1))));

C_glob = cov(double(eeg_data)');
C_glob = (C_glob + C_glob') / 2;
[Vg, Dg] = eig(C_glob);
[ev_g, ord_g] = sort(abs(diag(Dg)), 'descend');
Vg = Vg(:, ord_g);
idx_mid      = max(2, round(0.20 * N_chans)):max(3, round(0.35 * N_chans));
idx_tail80   = max(2, round(0.80 * N_chans)):max(2, N_chans - 1);
idx_tail85   = max(2, round(0.85 * N_chans)):max(2, N_chans - 1);
log_tail     = log10(max(median(ev_g(idx_tail80)) / max(median(ev_g(1:SSI_top_PCs)), 1e-15), 1e-15));
log_mid_tail = log10(max(median(ev_g(idx_tail85)) / max(median(ev_g(idx_mid)), 1e-15), 1e-15));
head_mid     = log10(max(median(ev_g(1:min(2, N_chans))) / max(median(ev_g(idx_mid)), 1e-15), 1e-15));
dip_frac     = trace(evecs_lead' * C_glob * evecs_lead) / max(trace(C_glob), 1e-15);
idx_vmid     = min(N_chans, 3):max(min(N_chans, 3), min(N_chans - 1, round(0.40 * N_chans)));
V_mid        = Vg(:, idx_vmid);
C_mid        = V_mid * diag(ev_g(idx_vmid)) * V_mid';
dip_mid      = trace(evecs_lead' * C_mid * evecs_lead) / max(trace(C_mid), 1e-15);
dip_eff      = 0.5 * (dip_frac + dip_mid);

epoch_samples = max(2, round(srate * 1.0));
n_epochs = floor(size(eeg_data, 2) / epoch_samples);
if n_epochs < 1
    m_eff = 3.0;
    return;
end
X_ep = reshape(eeg_data(:, 1:(n_epochs * epoch_samples)), N_chans, epoch_samples, n_epochs);
Eval_sub = zeros(N_chans, N_chans, n_epochs);
Evec_sub = zeros(N_chans, N_chans, n_epochs);
for epo = 1:n_epochs
    C_ep = cov(X_ep(:, :, epo)');
    C_ep = (C_ep + C_ep') / 2;
    c_tr = trace(C_ep) / N_chans;
    if c_tr > 0
        C_ep = (1 - 1e-9) * C_ep + (1e-9 * c_tr) * refCOV_reg;
    end
    [Evec_sub(:, :, epo), Eval_sub(:, :, epo)] = eig(C_ep, refCOV_reg, 'chol');
end
[m_eff, c1_bb] = local_compute_z3_noise_multiplier(Eval_sub, Evec_sub, refCOV_reg, evecs_Template_cov, SSI_top_PCs, -4);

% Slow-band (2-12 Hz) GEVD participation ratio and dipolar cosine alignment
eeg_slow = movmean(double(eeg_data), max(1, round(srate / 12)), 2) - movmean(double(eeg_data), max(1, round(srate / 2)), 2);
X_ep_s   = reshape(eeg_slow(:, 1:(n_epochs * epoch_samples)), N_chans, epoch_samples, n_epochs);
pr_s = zeros(1, n_epochs);
c1_s = zeros(1, n_epochs);
pk_s = zeros(1, n_epochs);
for epo = 1:n_epochs
    C_s = cov(X_ep_s(:, :, epo)');
    C_s = (C_s + C_s') / 2;
    c_tr = trace(C_s) / N_chans;
    if c_tr > 0
        C_s = (1 - 1e-9) * C_s + (1e-9 * c_tr) * refCOV_reg;
    end
    [Ec_s, Ev_s] = eig(C_s, refCOV_reg, 'chol');
    ev_r = abs(double(diag(Ev_s)));
    [ev_s, ord_s] = sort(ev_r, 'ascend');
    pk_s(epo) = ev_s(end);
    ev_a = ev_s(2:end);
    pr_s(epo) = (sum(ev_a)^2) / max(sum(ev_a.^2), 1e-12);
    vt_s = double(Ec_s(:, ord_s(end)));
    pt_s = refCOV_reg * vt_s;
    c1_s(epo) = norm(evecs_Template_cov' * (pt_s / max(norm(pt_s), 1e-12)));
end
tmask_s = (pk_s >= median(pk_s));
if ~any(tmask_s), tmask_s = true(1, n_epochs); end
PR_slow = mean(pr_s(tmask_s));
c1_eeg  = 0.5 * c1_bb + 0.5 * max(0.15, min(0.95, mean(c1_s(tmask_s))));

w_dip_mid  = 1.0 ./ (1.0 + exp(-(max(dip_mid, dip_eff) - 0.72) / 0.020));
w_head     = 1.0 ./ (1.0 + exp(-(head_mid - 2.15) / 0.10));
w_pr_slow  = 1.0 ./ (1.0 + exp((PR_slow - 2.25) / 0.18));
w_c1_low   = 1.0 ./ (1.0 + exp((c1_eeg - 0.52) / 0.04));
w_macro    = (1.0 - w_dip_mid) * w_head * w_pr_slow * w_c1_low;

w_tail     = 1.0 ./ (1.0 + exp((log_tail + 3.85) / 0.10));
w_mid_tail = 1.0 ./ (1.0 + exp((log_mid_tail + 1.85) / 0.08));
w_dip      = 1.0 ./ (1.0 + exp(-(dip_eff - 0.55) / 0.020));
w_rank     = w_tail * w_mid_tail * w_dip * (1.0 - w_macro);
end

function [m_eff, c1, PR] = local_compute_z3_noise_multiplier(Eval_sub, Evec_sub, refCOV_reg, evecs_Template_cov, SSI_top_PCs, minThreshold)
[N_chans, ~, w_len] = size(Eval_sub);
M = max(1, SSI_top_PCs);
if N_chans <= 2 || w_len < 1
    m_eff = double(M);
    c1    = 0.67;
    PR    = double(M);
    return;
end

all_evals_mat = zeros(N_chans, w_len);
part_ratios   = zeros(1, w_len);
top1_cos      = zeros(1, w_len);
Template_guess = double(evecs_Template_cov);
refCOV_reg_d   = double(refCOV_reg);

for epo = 1:w_len
    ev_raw = abs(double(diag(Eval_sub(:, :, epo))));
    [ev, ord] = sort(ev_raw, 'ascend');
    all_evals_mat(:, epo) = ev;
    ev_act = ev(2:end);
    part_ratios(epo) = (sum(ev_act)^2) / max(sum(ev_act.^2), 1e-12);
    
    v_top1 = double(Evec_sub(:, ord(end), epo));
    p_top1 = refCOV_reg_d * v_top1;
    top1_cos(epo) = norm(Template_guess' * (p_top1 / max(norm(p_top1), 1e-12)));
end

peak_ev = all_evals_mat(end, :);
topq_mask = (peak_ev >= median(peak_ev));
if ~any(topq_mask)
    topq_mask = true(1, w_len);
end

k_start = max(2, N_chans - (M^3) + 1);
active_evals = all_evals_mat(k_start:end, topq_mask);
ev_floor = 1e-6 * median(peak_ev(topq_mask));
pos_evals = log(active_evals(active_evals > ev_floor));
if numel(pos_evals) >= 2
    iqr_log = max(0.5, prctile(pos_evals, 75) - prctile(pos_evals, 25));
else
    iqr_log = 1.0;
end

PR = mean(part_ratios(topq_mask));
c1 = max(0.15, min(0.95, mean(top1_cos(topq_mask))));
M_d = double(M);

if minThreshold == -4
    scale_factor = double(N_chans) / (M_d^2);
    excess_iqr = max(1.0, iqr_log / (M_d - 1.0));
    m_raw = (scale_factor * PR) / ((iqr_log^2) * c1 * (excess_iqr^2));
    m_floor = max(M_d / max(M_d - 1.0, 1.0), (M_d - 1.0) / (excess_iqr^3));
    m_eff = max(m_floor, min(1.5 * M_d, m_raw));
else
    c1_floor = (M_d - 1.0) / M_d;
    w_nondip = min(1.0, max(0.0, (c1_floor - c1) / (1.0 - c1_floor)));
    iqr_ref = M_d - w_nondip;
    excess_iqr = 1.0 + max(0.0, iqr_log - iqr_ref) / (M_d - 1.0);
    m_raw = (M_d * PR) / ((iqr_log^2) * max(c1, c1_floor) * (excess_iqr^2));
    m_floor = max(M_d / max(M_d - 1.0, 1.0), M_d / (excess_iqr^3));
    m_eff = max(m_floor, min(1.5 * M_d, m_raw));
end
end

function out = stitch_blocks_cosine(b1, b2, b3, fade_len)
if fade_len < 2
    out = [b1, b2, b3];
    return;
end
w = 0.5 * (1 - cos(pi * (0:(fade_len - 1)) / (fade_len - 1)));
b12 = [b1(:, 1:end-fade_len), bsxfun(@times, b1(:, end-fade_len+1:end), 1 - w) + bsxfun(@times, b2(:, 1:fade_len), w), b2(:, fade_len+1:end)];
out = [b12(:, 1:end-fade_len), bsxfun(@times, b12(:, end-fade_len+1:end), 1 - w) + bsxfun(@times, b3(:, 1:fade_len), w), b3(:, fade_len+1:end)];
end
