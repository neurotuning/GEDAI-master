% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% GNU Affero General Public License v3
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License version 3 
% as published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Affero General Public License for more details.
%
% You should have received a copy of the GNU Affero General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [weights] = create_cosine_weights(channels, srate, epoch_size, fullshift)
%cosine weights to avoid overlaps for seamless re-contruction 

cosine_weights=zeros(channels,srate*epoch_size);

    %creating the weights (depending on odd or even)
    if(fullshift) %even
        for e=1:channels %for each electrode
            for u=1:srate*epoch_size %for each sample in the epoch
                cosine_weights(e,u) = 0.5-0.5*cos(2*u*pi/(srate*epoch_size));
            end
        end
    else %odd
        for e=1:channels
            for u=1:srate*epoch_size-1
                cosine_weights(e,u) = 0.5-0.5*cos(2*u*pi/(srate*epoch_size-1));
            end
        end
    end
    weights = cosine_weights;
end
