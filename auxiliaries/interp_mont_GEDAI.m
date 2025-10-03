% EEG = interp_mont() - Interpolate current data to coordinates from a specified
%               sfp file.
%
% Usage:
%   >>  com= interp_mont(EEG, coordfname, varargin);
%
% Inputs:
%   EEG         - input EEG structure
%   coordfname  - name of sfp file containing new channel coordinates.
%   varargin    - key/val pairs. See Options.
%
% Options:
%   nfids        - Number of coordinates at the begining of the coordinate file that
%                  should be treated as fiducials.
%   coreglndmrks - Cell array of channel labels to be used as input to the
%                  the 'warp' option in coregister.m. See help coregister.
%   manual       - ['on'|'off'] 
% Outputs:
%   EEG       - EEG structure updated with new coordinates.
%               input to 'manual' option of coregister.m. See help coregister.
% See also:
%   pop_interpmont

% Copyright (C) <2010>  <James Desjardins>
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function EEG = interp_mont_GEDAI(EEG,coordfname,varargin)


% Handle optional inputs.
g = struct(varargin{:});

try g.nfids;        catch, g.nfids        = 0;       end;
try g.coreglndmrks; catch, g.coreglndmrks = [];      end;
try g.manual;       catch, g.manual       = 'off';    end;


% warp current montage to coordinate file if coreglndmrks is specified.
if ~isempty(g.coreglndmrks)||strcmp(g.manual,'on');
    if ~isempty(g.coreglndmrks);
        eval(['coreglndmrks=', g.coreglndmrks ]);
    else
        coreglndmrks={};
    end
    [newlocs transform] = coregister(EEG.chanlocs, ...
				coordfname, ...
				'chaninfo1',EEG.chaninfo, ...
				'warp', coreglndmrks, ...
                'manual', g.manual);
            
     ndatch=length(EEG.chanlocs);
     nndatch=size(newlocs.pnt,1)-ndatch;
     
     for i=1:ndatch;
         EEG.chanlocs(i).X=newlocs.pnt(i,1);
         EEG.chanlocs(i).Y=newlocs.pnt(i,2);
         EEG.chanlocs(i).Z=newlocs.pnt(i,3);
     end
     
     for i=1:nndatch;
         EEG.chaninfo.nodatchans(i).X=newlocs.pnt(ndatch+i,1);
         EEG.chaninfo.nodatchans(i).Y=newlocs.pnt(ndatch+i,2);
         EEG.chaninfo.nodatchans(i).Z=newlocs.pnt(ndatch+i,3);
     end
     
     EEG = pop_chanedit(EEG,'convert','cart2all');
end


tmp.data=EEG.data;
tmp.chanlocs=EEG.chanlocs;

% disp(['Reading locations from ', ...
%        coordfname ', assuming first ', ...
%        num2str(g.nfids) ' coordinates are fiducials.']);
intrplocs=readlocs(coordfname);
EEG.chanlocs=intrplocs(g.nfids+1:length(intrplocs));

nintrplocs=length(intrplocs)-g.nfids;

EEG.nbchan=nintrplocs;
EEG.data=[];
EEG.data=zeros(EEG.nbchan,EEG.pnts,EEG.trials);

%EEG.chanlocs(1)
%length(EEG.chanlocs)

% Append 'tmp' to the begining of old channel labels.
for i=1:length(tmp.chanlocs);
    tmp.chanlocs(i).labels=['tmp',tmp.chanlocs(i).labels];
end

for i=1:length(tmp.chanlocs);
    EEG.chanlocs(EEG.nbchan+i).labels=tmp.chanlocs(i).labels;
    EEG.chanlocs(EEG.nbchan+i).X=tmp.chanlocs(i).X;
    EEG.chanlocs(EEG.nbchan+i).Y=tmp.chanlocs(i).Y;
    EEG.chanlocs(EEG.nbchan+i).Z=tmp.chanlocs(i).Z;
    EEG.chanlocs(EEG.nbchan+i).sph_theta=tmp.chanlocs(i).sph_theta;
    EEG.chanlocs(EEG.nbchan+i).sph_phi=tmp.chanlocs(i).sph_phi;
    EEG.chanlocs(EEG.nbchan+i).sph_radius=tmp.chanlocs(i).sph_radius;
    EEG.chanlocs(EEG.nbchan+i).theta=tmp.chanlocs(i).theta;
    EEG.chanlocs(EEG.nbchan+i).radius=tmp.chanlocs(i).radius;
    EEG.chanlocs(EEG.nbchan+i).type=tmp.chanlocs(i).type;
    if isfield(EEG.chanlocs,'urchan')
        if isfield(tmp.chanlocs,'urchan')
            EEG.chanlocs(EEG.nbchan+i).urchan=tmp.chanlocs(i).urchan;
        else
            EEG.chanlocs(EEG.nbchan+i).urchan=[];
        end
    end
    if isfield(EEG.chanlocs,'badchan')
        if isfield(tmp.chanlocs,'badchan')
            EEG.chanlocs(EEG.nbchan+i).badchan=tmp.chanlocs(i).badchan;
        else
            EEG.chanlocs.badchan=0;
        end
    end
    
    EEG.data(EEG.nbchan+i,:)=tmp.data(i,:);
end

EEG.nbchan=length(EEG.chanlocs);
EEG = eeg_interp_GEDAI(EEG, [1:nintrplocs],'spherical');
EEG = pop_select_GEDAI( EEG, 'channel',[1:nintrplocs] );

%if j>0
%    for i=size(labrep,1)
%        EEG.chanlocs(labrep{i,1}).labels=labrep{i,2};
%    end
%end


