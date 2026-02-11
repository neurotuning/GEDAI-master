function [C_OAS, rho] = oas_cov(X)
    % X: Input data (samples x channels)
    [N, P] = size(X);
    
    % 1. Compute Sample Covariance (S)
    S = cov(X);
    
    % 2. Define the Target Matrix (F)
    % F = mu * Eye, where mu is the average eigenvalue (trace/P)
    mu = trace(S) / P;
    
    % 3. Calculate Shrinkage Intensity (rho)
    % This is the "Oracle" approximation formula
    alpha = mean(diag(S.^2)) - (mu^2); % Variance of the sample covariance
    beta  = sum(sum(S.^2)) - sum(diag(S.^2)); % Off-diagonal squared sum
    
    % OAS specific rho formula
    rho = ((1 - 2/P) * sum(sum(S.^2)) + sum(diag(S))^2) / ...
          ((N + 1 - 2/P) * (sum(sum(S.^2)) - (sum(diag(S))^2)/P));
          
    % Constrain rho between 0 and 1
    rho = min(rho, 1);
    
    % 4. Compute Regularized Covariance
    C_OAS = (1 - rho) * S + rho * mu * eye(P);
end