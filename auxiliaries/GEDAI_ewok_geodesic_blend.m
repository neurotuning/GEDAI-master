function [C_gamma, tr_S] = GEDAI_ewok_geodesic_blend(C0, C1, gamma, reg_val)
% GEDAI_EWOK_GEODESIC_BLEND - Riemannian geodesic blending between two PSD
% covariance matrices based on McCann displacement interpolation (EWOK/Bures-Wasserstein).
%
% Inputs:
%   C0      - Baseline covariance (e.g. theoretical leadfield refCOV), (n x n)
%   C1      - Target empirical covariance (e.g. EC centroid), (n x n)
%   gamma   - Blending parameter:
%               gamma = 0  -> pure C0 (theoretical prior)
%               gamma = 1  -> pure C1 (empirical prior)
%               0 < gamma < 1 -> geodesic path
%   reg_val - Optional Tikhonov regularization fraction (default = 0.01)
%
% Outputs:
%   C_gamma - Blended covariance matrix, normalized to unit trace (Tr = 1)
%   tr_S    - Uhlmann root-fidelity (EWOK overlap) between C0 and C1

if nargin < 4 || isempty(reg_val)
    reg_val = 0.01;
end

n = size(C0, 1);

% Ensure symmetry
C0 = real((C0 + C0') / 2);
C1 = real((C1 + C1') / 2);

% Scale normalize inputs to unit trace
tr0 = trace(C0);
tr1 = trace(C1);
if tr0 > 0, C0 = C0 / tr0; end
if tr1 > 0, C1 = C1 / tr1; end

% Tikhonov regularization to ensure full-rank invertibility under average reference
C0_reg = (1 - reg_val) * C0 + (reg_val / n) * eye(n);
C1_reg = (1 - reg_val) * C1 + (reg_val / n) * eye(n);

% Spectral decomposition of C0_reg
[V0, D0] = eig((C0_reg + C0_reg') / 2);
d0 = max(diag(D0), 1e-12);
C0_sqrt    = V0 * diag(sqrt(d0)) * V0';
C0_invsqrt = V0 * diag(1 ./ sqrt(d0)) * V0';

% Uhlmann cross-term: S = (C0^(1/2) * C1 * C0^(1/2))^(1/2)
inner = C0_sqrt * C1_reg * C0_sqrt;
[Vi, Di] = eig((inner + inner') / 2);
di = max(diag(Di), 1e-12);
inner_sqrt = Vi * diag(sqrt(di)) * Vi';
tr_S = trace(inner_sqrt);

if gamma <= 0
    C_gamma = C0;
    return;
elseif gamma >= 1
    C_gamma = C1;
    return;
end

% Optimal transport operator: M = C0^(-1/2) * S * C0^(-1/2)
M = C0_invsqrt * inner_sqrt * C0_invsqrt;
M = real((M + M') / 2);

% McCann displacement interpolation operator
T_gamma = (1 - gamma) * eye(n) + gamma * M;

% Quadratic push-forward along the geodesic
C_raw = T_gamma * C0_reg * T_gamma;
C_raw = real((C_raw + C_raw') / 2);

% Project back to exact unit trace (EWOK manifold)
tr_raw = trace(C_raw);
if tr_raw > 0
    C_gamma = C_raw / tr_raw;
else
    C_gamma = C_raw;
end

end
