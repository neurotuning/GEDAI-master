function [optimalThreshold, maxSENSAIScore] = SENSAI_fminbnd(minThreshold, maxThreshold, EEGdata_epoched, srate, epoch_size, refCOV, Eval, Evec, noise_multiplier)

%   Finds the SENSAI maximum using Matlab's fminbnd.
%
%   Credits:  Tomas Ros & Abele Michela, NeuroTuning Lab

sensaifunc = @(artifactThreshold) SENSAIObjective(artifactThreshold, EEGdata_epoched, srate, epoch_size, refCOV, Eval, Evec, noise_multiplier);
options = optimset('Display', 'off', 'TolX', 1e-1);
[optimalThreshold, negMaxSENSAIScore] = fminbnd(sensaifunc, minThreshold, maxThreshold, options);
maxSENSAIScore = -negMaxSENSAIScore;

    function objective = SENSAIObjective(artifact_threshold, EEGdata_epoched, srate, epoch_size, refCOV, Eval, Evec, noise_multiplier_obj)
        % Compute the negative SENSAI score for the objective function
        [~, ~, SENSAI_score] = SENSAI(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec, noise_multiplier_obj);
        objective = -SENSAI_score;
    end
end