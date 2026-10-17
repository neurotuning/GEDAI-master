function [C_target, matched_labels] = GEDAI_load_empirical_centroid(chanlocs, band_name, reference_mode)
% GEDAI_LOAD_EMPIRICAL_CENTROID - Loads normative LEMON EC empirical covariance
% centroid, matches against current EEG channel montage, and applies reference.
%
% Inputs:
%   chanlocs       - EEG.chanlocs struct array
%   band_name      - Frequency band: 'Alpha' (default), 'Broadband', 'Theta', 'Beta', 'Delta', 'Gamma'
%   reference_mode - Reference mode (default: 'AvgRef')
%
% Outputs:
%   C_target       - Matched (nbchan x nbchan) covariance matrix, unit-trace normalized
%   matched_labels - Cell array of matched channel labels

if nargin < 2 || isempty(band_name), band_name = 'Alpha'; end
if nargin < 3 || isempty(reference_mode), reference_mode = 'AvgRef'; end

% Normalize band name
valid_bands = {'Broadband', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};
idx_b = find(strcmpi(band_name, valid_bands));
if ~isempty(idx_b)
    band_name = valid_bands{idx_b};
else
    band_name = 'Alpha';
end

aux_dir = fileparts(mfilename('fullpath'));
mat_path = fullfile(aux_dir, 'LEMON_EC_centroids.mat');
if ~exist(mat_path, 'file')
    % Fallback to checking GEDAI auxiliaries directory
    mat_path = fullfile(fileparts(which('GEDAI')), 'auxiliaries', 'LEMON_EC_centroids.mat');
end

if ~exist(mat_path, 'file')
    error('GEDAI:CentroidsNotFound', 'LEMON_EC_centroids.mat not found in %s', mat_path);
end

data = load(mat_path, 'centroids');
centroids = data.centroids;

C_raw = centroids.(band_name);
src_labels = lower(centroids.labels);
tar_labels = lower({chanlocs.labels});
nbchan = length(tar_labels);

idx_match = zeros(1, nbchan);
for i = 1:nbchan
    f = find(strcmp(tar_labels{i}, src_labels), 1);
    if isempty(f)
        % Partial match (e.g. without spaces or prefixes)
        f = find(contains(src_labels, tar_labels{i}) | contains(tar_labels{i}, src_labels), 1);
    end
    if ~isempty(f)
        idx_match(i) = f;
    end
end

matched_mask = (idx_match > 0);
matched_labels = {chanlocs(matched_mask).labels};

if all(matched_mask)
    C_target = C_raw(idx_match, idx_match);
elseif sum(matched_mask) >= 3
    % Submatrix for matched channels
    C_sub = C_raw(idx_match(matched_mask), idx_match(matched_mask));
    % Pad missing channels with diagonal variance from mean of matched
    C_target = eye(nbchan) * (trace(C_sub) / sum(matched_mask));
    C_target(matched_mask, matched_mask) = C_sub;
else
    error('GEDAI:MontageMismatch', 'Insufficient channel matches (%d / %d) with LEMON template.', sum(matched_mask), nbchan);
end

% Apply Average Reference Projector
R = eye(nbchan) - (1/nbchan)*ones(nbchan, nbchan);
C_target = R * C_target * R';
C_target = real((C_target + C_target') / 2);

% Scale normalize
tr = trace(C_target);
if tr > 0
    C_target = C_target / tr;
end

end
