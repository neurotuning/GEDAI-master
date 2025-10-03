% -Performs re-reference to average potential that is defined as the
%                           average potential across all the channels INCLUDING the initial
%                           reference channel which is usually unipolar-in that case, the average
%                           is calculated as sum(EEG.data,1)/(EEG.nbchan+1). For the theory
%                           behind this calculation, see the following reference paper.
%
% Reference
% Kim H, Luo J, Chu S, Cannard C, Hoffmann S, Miyakoshi M. (2023)
% ICA’s bug: How ghost ICs emerge from effective rank deficiency caused by EEG electrode interpolation and incorrect re-referencing
% Front. Signal Process. 2023 Apr 03; 3 DOI: 10.3389/frsip.2023.1064138

% Copyright (C) 2024 Makoto Miakoshi. Cincinnati Children's Hospital Medical Center. Makoto.Miyakoshi@cchmc.org
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
% THE POSSIBILITY OF SUCH DAMAGE.

% 01/24/2024 Makoto. Created. 

function EEG = GEDAI_nonRankDeficientAveRef(EEG)

% Subtract the average potential calculated including the initial reference electrode.
averagePotentialIncludingInitialReference = bsxfun(@minus, EEG.data, sum(EEG.data,1)/(EEG.nbchan+1));
EEG.data = averagePotentialIncludingInitialReference;

% Update the reference status.
EEG.ref = 'average';
for chIdx = 1:EEG.nbchan
    EEG.chanlocs(chIdx).ref = 'average';
end

% Run the checkset.
% EEG = eeg_checkset(EEG);

% Output eegh.
% com = sprintf('EEG = nonRankDeficientAveRef(EEG);');
% EEG = eegh(com, EEG);

% Report it is done.
% disp(sprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'))
% disp(sprintf('Re-reference to the average potential (including the initial reference) done.'))
% disp(sprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'))